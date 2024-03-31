#-----------------------------------------------------------------------------------------#
import sys
sys.path.append('./Libs')
import utilities as U
import class_functions as CF
#-----------------------------------------------------------------------------------------#
from pytorch_tabnet.tab_model import TabNetClassifier
import torch
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import pandas as pd
import numpy as np
import scipy
import os
#-----------------------------------------------------------------------------------------#

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

#-----------------------------------------------------------------------------------------#

csv_file = 'preprocessing/merged_data_with_labels.csv'
save_evaluation = 'evaluation'

#-----------------------------------------------------------------------------------------#

os.makedirs(save_evaluation, exist_ok=True)

#-----------------------------------------------------------------------------------------#

data = pd.read_csv(csv_file)
unique_values = data['zip_code'].unique()

# #-----------------------------------------------------------------------------------------#

feature_columns = ['Jan_rain', 'Feb_rain', 'Mar_rain', 'Apr_rain',
				'May_rain', 'Jun_rain', 'Jul_rain', 'Aug_rain', 'Sep_rain', 'Oct_rain',
				'Nov_rain', 'Dec_rain', 'Jan_temperature', 'Feb_temperature',
				'Mar_temperature', 'Apr_temperature', 'May_temperature',
				'Jun_temperature', 'Jul_temperature', 'Aug_temperature',
				'Sep_temperature', 'Oct_temperature', 'Nov_temperature',
				'Dec_temperature', 'Jan_wind_direction', 'Feb_wind_direction',
				'Mar_wind_direction', 'Apr_wind_direction', 'May_wind_direction',
				'Jun_wind_direction', 'Jul_wind_direction', 'Aug_wind_direction',
				'Sep_wind_direction', 'Oct_wind_direction', 'Nov_wind_direction',
				'Dec_wind_direction', 'Jan_windspeed', 'Feb_windspeed', 'Mar_windspeed',
				'Apr_windspeed', 'May_windspeed', 'Jun_windspeed', 'Jul_windspeed',
				'Aug_windspeed', 'Sep_windspeed', 'Oct_windspeed', 'Nov_windspeed',
				'Dec_windspeed']

# #-----------------------------------------------------------------------------------------#

max_epochs = 50 
patience = 15

clf = TabNetClassifier(
	n_d=64, n_a=64, n_steps=3,
	gamma=1.5, n_independent=2, n_shared=2,
	cat_idxs=[],
	cat_dims=[],
	cat_emb_dim=1,
	lambda_sparse=1e-4, momentum=0.3, clip_value=2.,
	optimizer_fn=torch.optim.Adam,
	optimizer_params=dict(lr=1e-4),
	scheduler_params = {"gamma": 0.95,
						"step_size": 20},
	device_name = device,
	scheduler_fn=torch.optim.lr_scheduler.StepLR, epsilon=1e-15
)

#-----------------------------------------------------------------------------------------#

total_loop = len(unique_values) - 1
evaluation_metrics = []
feature_importances = []

#-----------------------------------------------------------------------------------------#

for i in range(total_loop): 
	print(f"Loop {i + 1}/{total_loop}")
	val = unique_values[i]
	test = unique_values[i + 1] if i + 1 < len(unique_values) else unique_values[0]
	train = [x for j, x in enumerate(unique_values) if j != i and j != (i + 1)]

	train_data = data[data['zip_code'].isin(train)]
	val_data = data[data['zip_code'] == val]
	test_data = data[data['zip_code'] == test]
	
	X_train = train_data[feature_columns].values
	y_train = train_data['label'].values
	X_val = val_data[feature_columns].values
	y_val = val_data['label'].values
	X_test = test_data[feature_columns].values
	y_test = test_data['label'].values

	labels = y_train
	class_counts = np.bincount(labels, minlength=4)[1:]  # Adjusted to include minlength for safety
	total_instances = len(labels)
	weights = {}
	for class_label, class_label_count in enumerate(class_counts, start=1):  # Starting index from 1
		if class_label_count > 0:  # Check to ensure class_label_count is not zero
			weight = total_instances / (class_label_count * len(np.unique(labels)))
			weights[class_label] = weight
		else:  # Optionally, handle the zero instance case
			weights[class_label] = 0  # Assigning 0 or some default value to avoid 'inf'
	# print(f"Class weights: {weights}")

	clf.fit(
			X_train=X_train, y_train=y_train,
			eval_set=[(X_train, y_train), (X_val, y_val)],
			eval_name=['train', 'valid'],
			eval_metric=['accuracy'],
			max_epochs=max_epochs, patience=patience,
			batch_size=16, virtual_batch_size=32,
			num_workers=32,
			loss_fn=torch.nn.functional.cross_entropy,
			weights=weights,
			drop_last=True,
			augmentations=None  # aug, None
			) 

	preds = clf.predict(X_test)  # Use predict for multiclass to get the class directly
	precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, preds, average='weighted')
	accuracy = accuracy_score(y_test, preds)
	print(f"{test}: Precision={precision:.4f}, Recall={recall:.4f}, F1-score={f1_score:.4f}, Accuracy={accuracy:.4f}")

	evaluation_metrics.append({
							   'ZipCode': test,
							   'Precision': precision,
							   'Recall': recall,
							   'F1_Score': f1_score,
							   'Accuracy': accuracy
							  })

	# Extract and store feature importance if your model supports it
	if hasattr(clf, 'feature_importances_'):
		importance_dict = dict(zip(feature_columns, clf.feature_importances_))
		importance_dict['ZipCode'] = test  # Optionally add the zip code for reference
		feature_importances.append(importance_dict)
		print(f"Feature importance: {importance_dict}")

#-----------------------------------------------------------------------------------------#

# Convert evaluation metrics and feature importance to DataFrames
metrics_df = pd.DataFrame(evaluation_metrics)
importance_df = pd.DataFrame(feature_importances)

# Save the DataFrames to CSV files
metrics_csv_path = f'{save_evaluation}/evaluation_metrics.csv'
importance_csv_path = f'{save_evaluation}/feature_importance.csv'

metrics_df.to_csv(metrics_csv_path, index=False)
importance_df.to_csv(importance_csv_path, index=False)

print(f"Evaluation metrics successfully saved to {metrics_csv_path}")
print(f"Feature importance successfully saved to {importance_csv_path}")