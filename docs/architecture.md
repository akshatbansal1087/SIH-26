# System Architecture

## High-level flow

```text
             User
              |
              v
Satellite Data Archive (HDF5)
              |
              v
  +-----------------------+
  |                       |
  v                       v
Ch.0 (BT/IR Image)     Ch.2 (Raw Visible Image)
  |                       |
  v                       v
Branch A (Swin-T)      Branch B (EfficientNet-B0)
  |                       |
  +-----------+-----------+
              |
              v
       Late Fusion (MLP)
              |
              v
 Intensity (Estimated Wind Speed)
```

## Components

### Data Ingestion
Utilizes an optimized h5py DataLoader with persistent handles and multi-processing to ingest multidimensional satellite arrays and extract IR and Raw Visible channels.

### Branch A (Swin-T)
Utilizes a Swin Transformer (Tiny) to process Brightness Temperature (IR) satellite imagery and capture global thermal gradients.  

### Branch B (EfficientNet)
Employs an EfficientNet-B0 model to process RAW Visible satellite imagery to extract localized cloud structures and textures.  

### Late Fusion MLP
Processes the input data and generates a prediction.

### Optimization & Inference Engine
Handles hardware efficiency and error reduction by utilizing mixed-precision (FP16) training, Huber Loss for outlier resilience, and Test-Time Augmentation (TTA) during live inference to output exact continuous wind speed in knots.  
