import pickle

import pandas as pd
import numpy as np
import fire
import lightgbm as lgb

def main(
  train_data_path,
  valid_data_path,
  data_file_format,
  feature_cols,
  sample_weight_col,
  target_col,
  group_record_cols,
  eval_metric,
  eval_at,
  model_output_path,
<<<<<<< Updated upstream
=======
  model_output_metadata_path,
>>>>>>> Stashed changes
):

  if data_file_format == "csv":
    train = pd.read_csv(train_data_path)
    valid = pd.read_csv(valid_data_path)  
  elif data_file_format == "parquet":
    train = pd.read_parquet(train_data_path)
    valid = pd.read_parquet(valid_data_path)  
  elif data_file_format == "pickle":
    train = pd.read_pickle(train_data_path)
    valid = pd.read_pickle(valid_data_path)
  else:
    train = pd.read_csv(train_data_path)
    valid = pd.read_csv(valid_data_path)

  train = train.sort_values(group_record_cols)
  valid = valid.sort_values(group_record_cols)
  train_group=train.groupby(group_record_cols)[group_record_cols[0]].count().to_numpy()
  valid_group=valid.groupby(group_record_cols)[group_record_cols[0]].count().to_numpy()

  model = lgb.LGBMRanker(
    objective="lambdarank",
    boosting_type='gbdt', 
    metric=eval_metric,
    num_leaves=2**7-1,
    max_depth=7, 
    learning_rate=0.1, 
    n_estimators=100,
    early_stopping_rounds=10,
    random_state=42,
  )

  model.fit(
    X=train[feature_cols],
    y=train[target_col],
    sample_weight=train[sample_weight_col],
    group=train_group,
    eval_set=[(valid[feature_cols], valid[target_col])],
    eval_names=['valid'],
    eval_sample_weight=[valid[sample_weight_col]],
    eval_at=eval_at,
    eval_group=[valid_group],
    )

  with open(model_output_path, 'wb') as f:
    pickle.dump(model, f)

if __name__ == "__main__":
  fire.Fire(main)