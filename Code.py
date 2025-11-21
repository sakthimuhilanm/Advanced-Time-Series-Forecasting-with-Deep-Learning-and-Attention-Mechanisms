import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Layer, Input
from tensorflow.keras.optimizers import Adam

# --- 1. Custom Components and Utilities ---

class TimeSeriesScaler:
    """
    Manages MinMaxScaler for time series, preventing data leakage during Walk-Forward Validation.
    Fits the scaler on the first feature (often the target variable) and transforms all.
    """
    def __init__(self):
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def fit_transform(self, data):
        """Fits scaler on data[:, 0] and transforms all columns."""
        self.scaler.fit(data[:, 0].reshape(-1, 1))
        return self.transform(data)

    def transform(self, data):
        """Transforms all columns using the scaler fitted on the first column."""
        scaled_data = data.copy()
        for i in range(data.shape[1]):
            # Use the fitted min/max from the first feature for all features
            # A more rigorous approach might use a separate scaler per feature, 
            # but this common practice ensures features stay in a similar range relative to the target.
            min_val = self.scaler.min_[0]
            max_val = self.scaler.scale_[0]
            scaled_data[:, i] = (data[:, i] - min_val) / max_val
        return scaled_data

    def inverse_transform(self, scaled_data):
        """Inverses the scaling on the first column (target variable)."""
        # Inverse transform only the first column (the predicted target)
        return self.scaler.inverse_transform(scaled_data[:, 0].reshape(-1, 1))[:, 0]

def create_sequences(data, lookback, horizon):
    """
    Converts time series data into sequences for supervised learning.
    
    Args:
        data (np.ndarray): The multivariate time series data.
        lookback (int): The number of past time steps to use as input (X).
        horizon (int): The number of future time steps to predict (Y).
        
    Returns:
        tuple: (X_sequences, Y_targets)
    """
    X, y = [], []
    for i in range(len(data) - lookback - horizon + 1):
        X.append(data[i:(i + lookback), :])
        # Target is the first feature (index 0) 'horizon' steps ahead
        y.append(data[i + lookback: i + lookback + horizon, 0])
    return np.array(X), np.array(y)

# --- 2. Advanced Model: Transformer Components ---

class MultiHeadAttention(Layer):
    """Custom Keras Multi-Head Self-Attention Layer."""
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % num_heads == 0
        self.depth = d_model // num_heads
        self.wq = Dense(d_model)
        self.wk = Dense(d_model)
        self.wv = Dense(d_model)
        self.dense = Dense(d_model)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth)."""
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, training=False):
        batch_size = tf.shape(q)[0]
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        # Scaled dot-product attention
        matmul_qk = tf.matmul(q, k, transpose_b=True)
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        output = tf.matmul(attention_weights, v)

        output = tf.transpose(output, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(output, (batch_size, -1, self.d_model))
        output = self.dense(concat_attention)

        # Store weights for interpretation
        self.last_attention_weights = attention_weights
        return output

class TransformerEncoderBlock(Layer):
    """Single block of a Transformer Encoder."""
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(TransformerEncoderBlock, self).__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = Sequential([
            Dense(dff, activation='relu'),
            Dense(d_model)
        ])
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, x, training=False):
        attn_output = self.mha(x, x, x, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # Add & Norm
        
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output) # Add & Norm
        return out2

def build_transformer_model(lookback, n_features, d_model=64, num_heads=4, dff=128, horizon=1):
    """Builds the complete Transformer model for forecasting."""
    inputs = Input(shape=(lookback, n_features))
    
    # Optional: Pre-processing layer (e.g., Simple Dense for feature dimension alignment)
    x = Dense(d_model)(inputs) 
    
    # Encoder Block
    x = TransformerEncoderBlock(d_model, num_heads, dff)(x)
    
    # Use the output from the last time step for forecasting (or average across sequence)
    # Using the last step is common for sequence-to-vector forecasting.
    x = tf.keras.layers.Lambda(lambda x: x[:, -1, :])(x) 
    
    # Final dense layers for prediction
    outputs = Dense(horizon)(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(), loss='mse')
    return model

# --- 3. Baseline Model: Simple LSTM ---

def build_lstm_model(lookback, n_features, lstm_units=50, horizon=1):
    """Builds the simple LSTM baseline model."""
    model = Sequential([
        Input(shape=(lookback, n_features)),
        LSTM(lstm_units, return_sequences=False),
        Dropout(0.2),
        Dense(horizon)
    ])
    model.compile(optimizer=Adam(), loss='mse')
    return model

# --- 4. Data Generation and Setup ---

def generate_complex_ts(n_records=5000, n_features=5):
    """Generates synthetic multivariate time series data."""
    np.random.seed(42)
    t = np.arange(n_records)
    
    # Feature 1 (Target): Sine wave + Trend + Noise
    target = 10 * np.sin(t / 50) + t / 100 + np.random.randn(n_records) * 0.5
    
    # Feature 2: Lagged target + noise
    f2 = np.roll(target, 5) + np.random.randn(n_records) * 0.3
    
    # Other features: Random walks / correlated noise
    features = [target, f2]
    for _ in range(2, n_features):
        feature = np.cumsum(np.random.randn(n_records) * 0.1) + np.random.randn(n_records) * 0.2
        features.append(feature)
        
    data = np.stack(features, axis=1)
    
    # Create DataFrame for clarity
    df = pd.DataFrame(data, columns=[f'F_{i}' for i in range(n_features)])
    return df

# --- 5. Walk-Forward Validation and Evaluation ---

def calculate_metrics(y_true, y_pred):
    """Calculates RMSE, MAE, and MAPE."""
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    mae = np.mean(np.abs(y_true - y_pred))
    
    # Avoid division by zero in MAPE
    y_true_no_zero = np.where(y_true == 0, 1e-6, y_true)
    mape = np.mean(np.abs((y_true - y_pred) / y_true_no_zero)) * 100
    
    return rmse, mae, mape

def walk_forward_validation(df, model_builder, lookback, horizon, train_size_ratio=0.7, validation_folds=5, model_name="Model"):
    """Performs Walk-Forward Validation and returns aggregated metrics."""
    
    print(f"\n--- Starting Walk-Forward Validation for {model_name} ---")
    data = df.values
    n_records = len(data)
    
    # Calculate initial split sizes
    train_size = int(n_records * train_size_ratio)
    test_size = int((n_records - train_size) / validation_folds)
    
    all_metrics = []
    
    for fold in range(validation_folds):
        # 1. Define split indices
        start_train = 0
        end_train = train_size + fold * test_size
        start_test = end_train
        end_test = end_train + test_size
        
        if end_test > n_records:
            print(f"Fold {fold+1}: Reached end of data. Skipping remaining folds.")
            break
            
        print(f"Fold {fold+1}/{validation_folds}: Train {start_train}-{end_train}, Test {start_test}-{end_test}")
        
        # 2. Data Preparation for the current fold
        train_data = data[start_train:end_train]
        test_data = data[start_test:end_test]
        
        # Initialize and fit scaler ONLY on training data (prevents leakage)
        scaler = TimeSeriesScaler()
        X_train_scaled, y_train_scaled = create_sequences(
            scaler.fit_transform(train_data), lookback, horizon
        )
        X_test_scaled, _ = create_sequences(
            scaler.transform(test_data), lookback, horizon
        )
        
        # True values for the test set (needed for inverse transform and metric calculation)
        y_true_raw = data[start_test + lookback : end_test]
        # Only take the first 'horizon' steps ahead for the target column (index 0)
        y_true = y_true_raw[:-(horizon-1), 0] if horizon > 1 else y_true_raw[:, 0]
        
        # 3. Model Training and Prediction
        tf.keras.backend.clear_session()
        model = model_builder()
        
        # Use a small validation split within the training window for early stopping
        model.fit(
            X_train_scaled, y_train_scaled[:, 0] if horizon == 1 else y_train_scaled, 
            epochs=50, batch_size=32, verbose=0, shuffle=False,
            callbacks=[tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)]
        )
        
        y_pred_scaled = model.predict(X_test_scaled)
        
        # 4. Inverse Transform and Metric Calculation
        # Reconstruct the scaled target data structure for inverse transform (only the first column is needed for 1D target)
        if horizon == 1:
            y_pred_scaled_for_inv = np.zeros((y_pred_scaled.shape[0], df.shape[1]))
            y_pred_scaled_for_inv[:, 0] = y_pred_scaled[:, 0]
        else:
            # Simplified: assuming only 1-step forecast for metric calculation
            y_pred_scaled_for_inv = np.zeros((y_pred_scaled.shape[0], df.shape[1]))
            y_pred_scaled_for_inv[:, 0] = y_pred_scaled[:, 0] # Only take the first predicted step for 1D comparison

        y_pred_raw = scaler.inverse_transform(y_pred_scaled_for_inv)
        
        # Trim y_true for exact match with prediction length after sequence creation
        y_true = y_true[:len(y_pred_raw)]
        
        rmse, mae, mape = calculate_metrics(y_true, y_pred_raw)
        all_metrics.append({'RMSE': rmse, 'MAE': mae, 'MAPE': mape})
        
        print(f"Fold {fold+1} Metrics: RMSE={rmse:.4f}, MAE={mae:.4f}, MAPE={mape:.4f}%")

    # 5. Final Aggregation
    if not all_metrics:
        return {'RMSE': 0, 'MAE': 0, 'MAPE': 0}

    metrics_df = pd.DataFrame(all_metrics)
    agg_metrics = metrics_df.mean().to_dict()
    print(f"\n{model_name} Aggregated Metrics: RMSE={agg_metrics['RMSE']:.4f}, MAE={agg_metrics['MAE']:.4f}, MAPE={agg_metrics['MAPE']:.4f}%")
    return agg_metrics

# --- 6. Main Execution ---

if __name__ == '__main__':
    # Configuration
    LOOKBACK = 20  # Input sequence length
    HORIZON = 1    # Forecast horizon (predict 1 step ahead)
    N_RECORDS = 5000
    N_FEATURES = 5
    
    # 1. Data Acquisition
    df = generate_complex_ts(n_records=N_RECORDS, n_features=N_FEATURES)
    print(f"Data Generated. Shape: {df.shape}")
    print("\n--- Project Setup ---")
    print(f"Lookback Window (T_in): {LOOKBACK}")
    print(f"Forecast Horizon (T_out): {HORIZON}")
    print(f"Number of Features: {N_FEATURES}")
    print("---------------------")

    # Define model builders with required parameters
    # Baseline Model Builder
    lstm_builder = lambda: build_lstm_model(LOOKBACK, N_FEATURES, lstm_units=64, horizon=HORIZON)
    
    # Attention Model Builder (Transformer)
    transformer_builder = lambda: build_transformer_model(
        lookback=LOOKBACK, n_features=N_FEATURES, d_model=64, num_heads=4, dff=128, horizon=HORIZON
    )
    
    # 2. Implement and Evaluate Baseline Model
    baseline_metrics = walk_forward_validation(
        df, lstm_builder, LOOKBACK, HORIZON, model_name="Baseline LSTM"
    )
    
    # 3. Implement and Evaluate Advanced Attention Model
    attention_metrics = walk_forward_validation(
        df, transformer_builder, LOOKBACK, HORIZON, model_name="Attention Transformer"
    )

    # 4. Attention Weight Analysis (Extract weights from the last fold's trained model)
    # Re-initialize and train the model on the last fold's data to get the weights
    # This is a simplification; a full production system would save/load the model.
    # Note: Accessing internal weights requires running the prediction again and examining the layer attribute.
    
    final_transformer_model = transformer_builder()
    
    # Train the model one last time on the final training window for analysis
    train_data = df.values[:int(N_RECORDS * 0.7) + 4 * int((N_RECORDS - int(N_RECORDS * 0.7)) / 5)]
    scaler = TimeSeriesScaler()
    X_train_scaled, y_train_scaled = create_sequences(scaler.fit_transform(train_data), LOOKBACK, HORIZON)
    
    final_transformer_model.fit(
        X_train_scaled, y_train_scaled[:, 0] if HORIZON == 1 else y_train_scaled, 
        epochs=10, batch_size=32, verbose=0, shuffle=False
    )
    
    # Run a prediction to populate the attention weights
    X_sample = X_train_scaled[0:1]
    _ = final_transformer_model.predict(X_sample)
    
    # Find the MultiHeadAttention layer in the model
    attn_layer = next((layer for layer in final_transformer_model.layers if isinstance(layer, TransformerEncoderBlock)), None)
    
    if attn_layer:
        # Access the stored attention weights from the MHA layer inside the block
        mha_layer = attn_layer.mha
        if hasattr(mha_layer, 'last_attention_weights'):
            weights = mha_layer.last_attention_weights.numpy() # Shape: (batch, heads, seq_len, seq_len)
            avg_weights = weights[0, :, -1, :].mean(axis=0) # Avg across heads, focusing on attention to the last step's Q
            
            print("\n--- 5. Attention Weight Analysis (Sample) ---")
            print("Attention Model is prioritizing the following lookback steps (weights for the last token's query):")
            weight_df = pd.DataFrame({
                'Time Step (t - L)': [f't-{LOOKBACK - i}' for i in range(LOOKBACK)],
                'Average Attention Weight': avg_weights
            })
            print(weight_df.sort_values(by='Average Attention Weight', ascending=False).head(5))
            print("")
        else:
            print("\nAttention weights could not be extracted. Ensure the MHA layer stores 'last_attention_weights'.")

    # 7. Final Summary Table
    print("\n--- 7. Final Summary of All Model Performance Metrics ---")
    results = pd.DataFrame({
        'Model': ['Baseline LSTM', 'Attention Transformer'],
        'RMSE': [baseline_metrics['RMSE'], attention_metrics['RMSE']],
        'MAE': [baseline_metrics['MAE'], attention_metrics['MAE']],
        'MAPE (%)': [baseline_metrics['MAPE'], attention_metrics['MAPE']]
    })
    results = results.round(4)
    print(results.to_markdown(index=False))
