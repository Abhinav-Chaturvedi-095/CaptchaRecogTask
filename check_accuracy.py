from script import predict_captcha,load_data,load_model,get_device

csv_path = "/home/abhinav/TASK/dataset/captcha_data.csv"

# Load Model
model_path = "/home/abhinav/TASK/models_pth/crnn_model_199.pth"
device = get_device()
model = load_model(path=model_path,device=device)

# Run Predictions and Calculate Accuracy
correct = 0
total = 0

_,val_df,_ = load_data(csv_path=csv_path)

for index, row in val_df.iterrows():
    image_path = row["image_path"]
    true_solution = str(row["solution"]).strip()  # Convert to string & remove spaces

    predicted_solution = predict_captcha(model=model,image_path=image_path,device=device)
    print(f"true: {true_solution}, predicted: {predicted_solution}")
    if predicted_solution is not None:
        total += 1
        if predicted_solution == true_solution:
            correct += 1

# Compute Accuracy
accuracy = (correct / total) * 100 if total > 0 else 0
print(f"Model Accuracy: {accuracy:.2f}%")
