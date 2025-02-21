import TrainFunctions
import LoadData
import argparse
import LoadData

def parse_args():
    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument('--hyperparameter', '-hp', action='store_true', help='Perform hyperparameter optimization')
    parser.add_argument('--model', '-m', type=str, default="SiameseModel", help='Model name')
    parser.add_argument('--data', '-d', type=str, default="./data/benchmark.db", help='Path to the data file')
    parser.add_argument('--zero-weight', '-weight', type=float, help='Weight for zero alignment score')
    parser.add_argument('--overwrite', '-w', action='store_true', help='Overwrite the existing data')
    parser.add_argument('--epochs', '-e', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for training')
    parser.add_argument('--metrics', type=str, default='mse, mae, mape', help='Metrics to use for training')
    parser.add_argument('--output', '-o', type=str, default="./log/", help='Output directory for logs and models')
    return parser.parse_args()

# Main Function
if __name__ == "__main__":
    args = parse_args()

    # Script Arguments
    DATAPATH = args.data
    TRAINHYPERPARAMETER = True if args.hyperparameter else False
    MODEL = args.model
    OVERWRITE = True if args.overwrite else False
    ZERO_WEIGHT = args.zero_weight
    print(f"Zero Weight from Input: {ZERO_WEIGHT}")

    # Model Related Parameters
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    LEARNING_RATE = args.learning_rate
    METRICS = TrainFunctions.SplitString(args.metrics)

    # ===== Loading the Model =====
    Models = ['SiameseModel']   # Update this list with new models
    print(f"Attempting to load {MODEL}")
    if args.model == "SiameseModel":
        library = 'tensorflow'
        from models.SiameseModel import get_model
        from models.SiameseModel import HyperParameterTraining
    else:
        print(f"ERROR: {args.model} is not a valid model")
        exit(1)

    # ===== Load The Data =====
    if library == 'tensorflow':
        # train_set, val_set, test_set = TrainFunctions.LoadDataTensorflow(DATAPATH, overwrite=OVERWRITE, zero_weight=ZERO_WEIGHT)
        train_set, val_set, test_set = LoadData.CreateTensorflowDataset(DATAPATH, overwrite=OVERWRITE, zero_weight=ZERO_WEIGHT)
    else:
        raise NotImplementedError(f"LoadData for {library} is not implemented yet")

    # ===== Verify Path Validity =====
    PATH = TrainFunctions.CheckPath(args.output, library, MODEL)
    print(f"Checking path validity for {PATH}")
    if not PATH:
        print(f"ERROR: {PATH} is not a valid path")
        exit(1)


    if TRAINHYPERPARAMETER:
        # ===== Hyperparameter Optimization =====
        print(f"===== HYPERPARAMETER TRAINING =====")
        bestparams, bestmodel = HyperParameterTraining(train_set, val_set, metrics=METRICS, n_trials=1, bestModelPath=PATH)
        print(f"Best Parameters: {bestparams}")

        # ===== Evaluate the Best Model =====
        TrainFunctions.EvaluateModel(bestmodel, DATAPATH, METRICS, BATCH_SIZE)
    else:
        # ===== Train the Tensorflow Model =====
        print(f"Training for {args.epochs} epochs with batch size {args.batch_size} and learning rate {args.learning_rate}")
        if (library == 'tensorflow'):
            model = get_model(learning_rate=LEARNING_RATE, metrics=METRICS)
            model, history = TrainFunctions.TrainTensorflowModel(model, train_set, val_set, epochs=EPOCHS, batch_size=BATCH_SIZE)

            # Dump the model and the history
            TrainFunctions.DumpModel(model, PATH, library, history.history)

            # ===== Evaluate the Best Model =====
            TrainFunctions.EvaluateModel(model, DATAPATH, METRICS, BATCH_SIZE)

        else:
            raise NotImplementedError("Training for pytorch is not implemented yet")
