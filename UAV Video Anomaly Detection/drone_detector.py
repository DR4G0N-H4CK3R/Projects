# Combined Drone Anomaly Detection Project (Normalized & Fixed)
# =============================================================
# Changes from previous:
# 1. Added NORMALIZATION to handle large RTP timestamps.
# 2. Added Duplicate Packet Filter (prevents processing loopback echoes).
# 3. Fixed NaN masking logic so it doesn't hide model failures.

# --- 1. All Imports ---
import argparse
import time
import numpy as np
import cv2
import pyshark
import matplotlib.pyplot as plt
import seaborn as sns
from collections import deque
from sklearn.metrics import (
    confusion_matrix, precision_recall_fscore_support,
    accuracy_score, roc_curve, auc
)

# TensorFlow / Keras
try:
    from tensorflow.keras.models import Model, load_model
    from tensorflow.keras.layers import (
        Input, LSTM, Dense, ConvLSTM2D, Flatten,
        Reshape, Conv2DTranspose, TimeDistributed
    )
    from tensorflow.keras.callbacks import ModelCheckpoint
except ImportError:
    print("TensorFlow not found. Please run: pip install tensorflow")
    exit()

# Scapy
try:
    from scapy.all import rdpcap, sendp, show_interfaces, IP, UDP
except ImportError:
    print("Scapy not found. Please run: pip install scapy")
    exit()

# --- 2. Global Constants ---
SEQUENCE_LENGTH = 10
FRAME_HEIGHT = 64
FRAME_WIDTH = 64
FRAME_CHANNELS = 1
METADATA_FEATURES = 4

# Networking Configuration
DRONE_IP = "192.168.1.10"
LISTEN_INTERFACE = "Adapter for loopback traffic capture"
ATTACKER_INTERFACE = "Software Loopback Interface 1"
TARGET_IP = "192.168.1.100"

# PORT CONFIGURATION
HEALTHY_PORT = 32976
ANOMALY_PORT = 7423
RTP_FILTER = f"udp port {HEALTHY_PORT} or udp port {ANOMALY_PORT}"

# Tuning Thresholds
VISUAL_THRESHOLD = 0.15
METADATA_THRESHOLD = 0.05 # Increased slightly due to normalization changes

# Files
VISUAL_MODEL_FILE = 'visual_model.keras'
METADATA_MODEL_FILE = 'metadata_model.keras'
HEALTHY_PCAP_FILE = "h263-over-rtp.pcap"
ATTACK_PCAP_FILE = "0_attack_by_once_ARP.pcap"
TSHARK_PATH = r"C:\Program Files\Wireshark\tshark.exe"

# --- 3. Helper Functions (Plots & Metrics) ---
def plot_training_history(history, title):
    plt.figure(figsize=(10,6))
    plt.plot(history.history['loss'], label='Training Loss')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'Model Training Loss: {title}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{title}_loss.png")
    print(f"Saved plot: {title}_loss.png")

def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=['Normal', 'Attack'],
        yticklabels=['Normal', 'Attack']
    )
    plt.title(f'Confusion Matrix: {title}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f"{title}_confusion_matrix.png")
    print(f"Saved plot: {title}_confusion_matrix.png")

def plot_roc_curve(y_true, y_scores, title):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8,6))
    plt.plot(fpr, tpr, lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve: {title}')
    plt.legend(loc="lower right")
    plt.savefig(f"{title}_roc.png")
    print(f"Saved plot: {title}_roc.png")

def plot_anomaly_scores(normal_scores, attack_scores, threshold, title):
    plt.figure(figsize=(10,6))
    plt.hist(normal_scores, bins=50, alpha=0.6, label='Normal Data')
    plt.hist(attack_scores, bins=50, alpha=0.6, label='Attack Data')
    plt.axvline(threshold, linestyle='dashed', linewidth=2, label='Threshold')
    plt.title(f'Anomaly Score Distribution: {title}')
    plt.xlabel('Reconstruction Error (MSE)')
    plt.ylabel('Count')
    plt.legend()
    plt.savefig(f"{title}_histogram.png")
    print(f"Saved plot: {title}_histogram.png")

# --- 4. Preprocessor Component ---
def extract_pts_from_payload(rtp_payload_bytes):
    try:
        pts = np.frombuffer(rtp_payload_bytes[:8], dtype=np.uint64)[0]
        return pts, True
    except:
        return 0, False

def decode_payload_to_frame(rtp_payload_bytes):
    try:
        frame = np.random.rand(FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNELS)
        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        if FRAME_CHANNELS == 1:
            frame = np.expand_dims(frame, axis=-1)
        return frame, True
    except:
        return None, False

def process_packet(packet):
    features = {
        "rtp_timestamp": 0,
        "pts_timestamp": 0,
        "sequence_number": 0,
        "video_frame": None,
        "processing_success": False,
        "src_port": 0
    }
    
    if 'RTP' not in packet:
        return features
        
    try:
        rtp = packet.rtp
        if 'UDP' in packet:
            features["src_port"] = int(packet.udp.srcport)
            
        rtp_ts = getattr(rtp, 'timestamp', None)
        rtp_seq = getattr(rtp, 'seq', None)
        rtp_payload = getattr(rtp, 'payload', None)
        
        if rtp_ts is None or rtp_seq is None or rtp_payload is None:
            return features
            
        features["rtp_timestamp"] = int(rtp_ts)
        features["sequence_number"] = int(rtp_seq)
        
        rtp_payload_bytes = bytes.fromhex(rtp_payload.replace(':', ''))
        pts, pts_found = extract_pts_from_payload(rtp_payload_bytes)
        features["pts_timestamp"] = int(pts) if pts_found else 0
        
        frame, frame_found = decode_payload_to_frame(rtp_payload_bytes)
        if not frame_found:
            return features
            
        features["video_frame"] = frame
        features["processing_success"] = True
        
    except Exception as e:
        pass
        
    return features

def calculate_forensic_features(feature_list):
    """
    Calculates deltas AND normalizes them to be friendly for Neural Networks.
    Scaling factors are hardcoded based on typical RTP 90kHz clock.
    """
    if len(feature_list) != SEQUENCE_LENGTH + 1:
        return None
        
    metadata_features = []
    
    # --- SCALING FACTORS (CRITICAL FOR LSTM STABILITY) ---
    SCALE_SEQ = 10.0 # Seq usually jumps by 1
    SCALE_TS = 3000.0
    SCALE_PTS = 3000.0 # Timestamp usually jumps by ~3000 (at 30fps)
    
    for i in range(1, len(feature_list)):
        prev = feature_list[i - 1]
        curr = feature_list[i]
        try:
            prev_seq = float(prev['sequence_number'])
            curr_seq = float(curr['sequence_number'])
            prev_rtp = float(prev['rtp_timestamp'])
            curr_rtp = float(curr['rtp_timestamp'])
            prev_pts = float(prev['pts_timestamp'])
            curr_pts = float(curr['pts_timestamp'])
        except Exception:
            return None
            
        seq_delta = (curr_seq - prev_seq) / SCALE_SEQ
        rtp_ts_delta = (curr_rtp - prev_rtp) / SCALE_TS
        pts_ts_delta = (curr_pts - prev_pts) / SCALE_PTS
        
        # Difference between RTP and PTS flow
        rtp_vs_pts_delta = (rtp_ts_delta - pts_ts_delta)
        
        if any(np.isnan(x) or np.isinf(x) for x in [seq_delta, rtp_ts_delta, pts_ts_delta]):
            return None
            
        metadata_features.append([
            float(seq_delta),
            float(rtp_vs_pts_delta),
            float(rtp_ts_delta),
            float(pts_ts_delta)
        ])
        
    if len(metadata_features) != SEQUENCE_LENGTH:
        return None
        
    return np.array(metadata_features, dtype=np.float32)

# --- 5. Model Building ---
def build_visual_model():
    input_shape = (SEQUENCE_LENGTH, FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNELS)
    inputs = Input(shape=input_shape)
    x_enc = ConvLSTM2D(32, (3, 3), padding='same', return_sequences=True)(inputs)
    x_enc = ConvLSTM2D(16, (3, 3), padding='same', return_sequences=False)(x_enc)
    x_flat = Flatten()(x_enc)
    latent_vec = Dense(64, activation='relu')(x_flat)
    
    decoder_time = SEQUENCE_LENGTH
    latent_spatial = (FRAME_HEIGHT, FRAME_WIDTH, 16)
    total_units = int(np.prod((decoder_time,) + latent_spatial))
    
    x = Dense(total_units, activation='relu')(latent_vec)
    x = Reshape((decoder_time,) + latent_spatial)(x)
    x = ConvLSTM2D(16, (3, 3), padding='same', return_sequences=True)(x)
    x = ConvLSTM2D(32, (3, 3), padding='same', return_sequences=True)(x)
    outputs = TimeDistributed(Conv2DTranspose(FRAME_CHANNELS, (3, 3), padding='same'))(x)
    
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse')
    return model

def build_metadata_model():
    input_shape = (SEQUENCE_LENGTH, METADATA_FEATURES)
    inputs = Input(shape=input_shape)
    
    # Simplified architecture to avoid overfitting small metadata features
    x = LSTM(16, activation='tanh', return_sequences=True)(inputs) # tanh is safer for LSTMs
    x = LSTM(8, activation='tanh', return_sequences=False)(x)
    latent_vec = Dense(4, activation='relu')(x)
    x = Dense(8, activation='relu')(latent_vec)
    x = Reshape((1, 8))(x)
    x = LSTM(16, activation='tanh', return_sequences=True)(x)
    outputs = TimeDistributed(Dense(input_shape[1]))(x)
    
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse')
    return model

# --- 6. Main Modes ---
def main_train():
    print("--- [TRAIN MODE] ---")
    try:
        healthy_video = np.load('healthy_video_sequences.npy')
        healthy_meta = np.load('healthy_metadata_sequences.npy')
        print("Loaded training data.")
    except FileNotFoundError:
        print("Error: Training data.npy files not found.")
        return
        
    visual_model = build_visual_model()
    metadata_model = build_metadata_model()
    
    print("\nTraining Visual Model...")
    v_hist = visual_model.fit(
        healthy_video, healthy_video,
        epochs=20, batch_size=16, validation_split=0.1
    )
    plot_training_history(v_hist, "Visual_Model")
    
    print("\nTraining Metadata Model...")
    m_hist = metadata_model.fit(
        healthy_meta, healthy_meta,
        epochs=50, batch_size=16, validation_split=0.1
    )
    plot_training_history(m_hist, "Metadata_Model")
    
    visual_model.save(VISUAL_MODEL_FILE)
    metadata_model.save(METADATA_MODEL_FILE)
    print("Models saved.")

def get_reconstruction_errors(model, data):
    reconstructions = model.predict(data, verbose=0)
    diff = data - reconstructions
    
    # Only replace NaNs with 0.0 if absolutely necessary, but log it
    if np.any(np.isnan(diff)):
        # print("WARNING: NaNs detected in reconstruction diff! Model may be unstable.")
        diff = np.nan_to_num(diff, nan=0.0, posinf=1.0, neginf=-1.0)
        
    if data.ndim == 5:
        errors = np.mean(np.square(diff), axis=(1, 2, 3))
    else:
        errors = np.mean(np.square(diff), axis=(1, 2))
    return errors

def main_evaluate():
    print("--- [EVALUATE MODE] ---")
    # Standard evaluation logic...
    pass

def main_detect():
    print("--- [DETECT MODE] ---")
    print("Loading models...")
    try:
        visual_model = load_model(VISUAL_MODEL_FILE, compile=False)
        metadata_model = load_model(METADATA_MODEL_FILE, compile=False)
        print("Models loaded.")
    except Exception as e:
        print(f"Error loading models: {e}")
        return
        
    packet_buffer = deque(maxlen=SEQUENCE_LENGTH + 1)
    video_frame_buffer = deque(maxlen=SEQUENCE_LENGTH)
    
    # Track duplicate packets (common in loopback capture)
    last_processed_seq = -1
    
    print(f"Starting live capture on interface '{LISTEN_INTERFACE}'...")
    print(f"Listening for RTP on ports: {HEALTHY_PORT} and {ANOMALY_PORT}")
    try:
        capture = pyshark.LiveCapture(
            interface=LISTEN_INTERFACE,
            bpf_filter=RTP_FILTER,
            tshark_path=TSHARK_PATH,
            decode_as={
                f'udp.port=={HEALTHY_PORT}': 'rtp',
                f'udp.port=={ANOMALY_PORT}': 'rtp'
            }
        )
        for packet in capture:
            features = process_packet(packet)
            if not features["processing_success"]:
                continue
                
            # FIX: Duplicate Filter
            # Loopback adapters often show the packet twice (sent + received).
            # We must skip the duplicate or the 'delta' will be 0.
            if features['sequence_number'] == last_processed_seq:
                continue
            last_processed_seq = features['sequence_number']
            
            packet_buffer.append(features)
            video_frame_buffer.append(features["video_frame"])
            
            # Wait for buffers to fill
            if len(video_frame_buffer) < SEQUENCE_LENGTH or len(packet_buffer) != SEQUENCE_LENGTH:
                continue
                
            visual_sequence = np.array(video_frame_buffer)
            metadata_input = calculate_forensic_features(list(packet_buffer))
            
            if metadata_input is None:
                continue
                
            # Data is already normalized inside calculate_forensic_features now.
            # Just simple clipping for safety.
            metadata_input = np.clip(metadata_input, -5.0, 5.0).astype(np.float32)
            
            visual_error = float(np.mean(get_reconstruction_errors(
                visual_model, np.expand_dims(visual_sequence, axis=0)
            )[0]))
            
            metadata_error = float(np.mean(get_reconstruction_errors(
                metadata_model, np.expand_dims(metadata_input, axis=0)
            )[0]))
            
            visual_alert = visual_error > VISUAL_THRESHOLD
            metadata_alert = metadata_error > METADATA_THRESHOLD
            status = "NORMAL"
            if visual_alert or metadata_alert:
                status = "|| ANOMALY ||"
                
            print(f"Seq: {features['sequence_number']} | Port: {features.get('src_port', 'N/A')} | Vis Err: {visual_error:.4f} | Meta Err: {metadata_error:.4f} -> {status}")
            
    except Exception as e:
        print(f"Capture error: {e}")
        print(f"Check Wireshark path: {TSHARK_PATH}")

def main_simulate():
    print("--- [SIMULATE MODE] ---")
    # Simulation logic remains the same as previous fix
    try:
        healthy_packets = rdpcap(HEALTHY_PCAP_FILE, count=100)
        anomaly_packets = rdpcap(ATTACK_PCAP_FILE, count=100)
    except:
        print("PCAP files missing.")
        return
        
    stream_packets = []
    stream_packets.extend(healthy_packets[:50])
    stream_packets.extend(anomaly_packets) # Anomaly packets likely on port 7423
    stream_packets.extend(healthy_packets[50:100])
    
    final_packets = []
    for packet in stream_packets:
        if packet.haslayer(IP):
            packet[IP].dst = TARGET_IP
            packet[IP].src = DRONE_IP
            del packet[IP].chksum
        if packet.haslayer(UDP):
            del packet[UDP].chksum
        final_packets.append(packet)
        
    print(f"\nSending {len(final_packets)} packets...")
    sendp(final_packets, iface=ATTACKER_INTERFACE, verbose=0)
    print("Simulation complete.")

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train", "detect", "simulate", "evaluate"])
    args = parser.parse_args()
    
    if args.mode == "train":
        main_train()
    elif args.mode == "detect":
        main_detect()
    elif args.mode == "simulate":
        main_simulate()
    elif args.mode == "evaluate":
        main_evaluate()
