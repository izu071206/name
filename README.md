INFO:__main__:============================================================
INFO:__main__:IMPROVED TRAINING PIPELINE
INFO:__main__:============================================================
INFO:__main__:Loading processed data...
INFO:__main__:Train: (50, 553), Val: (6, 553), Test: (30, 553)
INFO:__main__:============================================================
INFO:__main__:TRAINING BASE MODELS
INFO:__main__:============================================================
INFO:__main__:Data Imbalance Ratio (Neg/Pos): 0.00
INFO:__main__:
[1/3] Training Random Forest...
INFO:src.models.random_forest_model:Training RandomForest...
INFO:src.models.random_forest_model:Validation accuracy: 1.0000
INFO:src.models.random_forest_model:Training accuracy: 1.0000
INFO:src.models.random_forest_model:Model saved to /home/izumi071/Documents/demo3/demo(fixed)/demo2/models/random_forest_model.pkl
INFO:__main__:✓ Random Forest training completed
INFO:__main__:
[2/3] Training XGBoost...
INFO:src.models.xgboost_model:Training XGBoost...
Traceback (most recent call last):
  File "/home/izumi071/Documents/demo3/demo(fixed)/demo2/./src/models/train_improved.py", line 335, in <module>
    main()
  File "/home/izumi071/Documents/demo3/demo(fixed)/demo2/./src/models/train_improved.py", line 295, in main
    base_models = train_base_models(X_train, y_train, X_val, y_val, config)
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/izumi071/Documents/demo3/demo(fixed)/demo2/./src/models/train_improved.py", line 115, in train_base_models
    xgb_model.train(X_train, y_train, X_val, y_val)
  File "/home/izumi071/Documents/demo3/demo(fixed)/demo2/src/models/xgboost_model.py", line 69, in train
    self.model.fit(
  File "/home/izumi071/Documents/demo3/demo(fixed)/demo2/.venv/lib/python3.12/site-packages/xgboost/core.py", line 774, in inner_f
    return func(**kwargs)
           ^^^^^^^^^^^^^^
  File "/home/izumi071/Documents/demo3/demo(fixed)/demo2/.venv/lib/python3.12/site-packages/xgboost/sklearn.py", line 1761, in fit
    raise ValueError(
ValueError: Invalid classes inferred from unique values of `y`.  Expected: [0], got [1]
loi nay fix the nao
