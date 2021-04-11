from pipeline.pipeline_model import Pipeline

print("EXTRACTING FEATURES")
pipeline = Pipeline()

print("TRANING MODEL")
pipeline.train_classification_model()

print("TESTING")
pipeline.test_classification_model()