To train & run the detections:
    step 1: create an virtualenv
    step 2: run the following command 
            $pip3 install -r requirements.txt
    step 3: To generate more augmented data, provide the dataset path and run following command:
            $python3 data_augmentation.py
    step 4: run the script by mentioning your path inside the script for dataset & saving model
            $python3 script.py
    Model will start training & save model for each epoch in the specified folder.
    step 5: To plot the Training Loss, Validation Loss and Validation Accuracy, provide the csv path and run following command:
        $python3 plot_graph.py

