INFO:__main__:============================================================
INFO:__main__:IMPROVED TRAINING PIPELINE - FIXED VERSION
INFO:__main__:============================================================
INFO:__main__:Loading processed data...
INFO:__main__:Train: (50, 553), Val: (6, 553), Test: (30, 553)
INFO:__main__:============================================================
INFO:__main__:TRAINING BASE MODELS
INFO:__main__:============================================================
INFO:__main__:
Training set class distribution:
INFO:__main__:  Class 1 (Obfuscated): 50 samples
ERROR:__main__:❌ Training set: Missing class 0 (Benign)!
INFO:__main__:
Validation set class distribution:
INFO:__main__:  Class 1 (Obfuscated): 6 samples
ERROR:__main__:❌ Validation set: Missing class 0 (Benign)!
ERROR:__main__:
============================================================
ERROR:__main__:CRITICAL ERROR: Training set is missing one or both classes!
ERROR:__main__:============================================================
ERROR:__main__:
Possible causes:
ERROR:__main__:1. Not enough samples in data/benign/ and data/obfuscated/
ERROR:__main__:2. Data split ratio is too extreme (check config/dataset_config.yaml)
ERROR:__main__:3. Samples failed to extract features
ERROR:__main__:
Solutions:
ERROR:__main__:1. Add more binary samples to data/benign/ and data/obfuscated/
ERROR:__main__:2. Check if you have at least 10-20 samples of EACH type
ERROR:__main__:3. Re-run: python main.py generate-dataset
ERROR:__main__:============================================================
ERROR:__main__:
============================================================
ERROR:__main__:TRAINING FAILED!
ERROR:__main__:============================================================
ERROR:__main__:Error: Cannot train with incomplete training set. Need both Benign and Obfuscated samples.
Traceback (most recent call last):
  File "/home/izumi071/Documents/demo3/demo(fixed)/demo2/./src/models/train_improved.py", line 741, in main
    base_models = train_base_models(X_train, y_train, X_val, y_val, config)
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/izumi071/Documents/demo3/demo(fixed)/demo2/./src/models/train_improved.py", line 472, in train_base_models
    raise ValueError("Cannot train with incomplete training set. Need both Benign and Obfuscated samples.")
ValueError: Cannot train with incomplete training set. Need both Benign and Obfuscated samples.
