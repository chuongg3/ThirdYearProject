import TrainFunctions
import LoadData

# Main Function
if __name__ == "__main__":
    args = TrainFunctions.parse_args()

    # ===== Loading the Model =====
    Models = ['SiameseModel']   # Update this list with new models
    MODEL = args.model
    print(f"Attempting to load {MODEL}")
    if args.model == "SiameseModel":
        library = 'tensorflow'
        from models.SiameseModel import get_model
    else:
        print(f"ERROR: {args.model} is not a valid model")
        exit(1)

    # ===== Load The Data =====
    overwrite = True if args.overwrite else False
    if library == 'tensorflow':
        train_set, val_set, test_set = TrainFunctions.LoadDataTensorflow(args.data, args.batch_size, overwrite=overwrite)
    else:
        raise NotImplementedError(f"LoadData for {library} is not implemented yet")

    # ===== Verify Path Validity =====
    PATH = TrainFunctions.CheckPath(args.output, library, MODEL)
    print(f"Checking path validity for {PATH}")
    if not PATH:
        print(f"ERROR: {PATH} is not a valid path")
        exit(1)

    # ===== Train the Tensorflow Model =====
    print(f"Training for {args.epochs} epochs with batch size {args.batch_size} and learning rate {args.learning_rate}")
    if (library == 'tensorflow'):
        model = get_model(learning_rate=args.learning_rate)
        model, history = TrainFunctions.TrainTensorflowModel(model, train_set, val_set, args.epochs, args.learning_rate)
        
        # Dump the model and the history
        TrainFunctions.DumpModel(model, PATH, library, history.history)

        # Test the Model
        print("===== ALL TEST DATA =====")
        data = TrainFunctions.EvaluateModel(model, test_set)
        print(f"Test Loss: {data[0]}")
        print(f"Test MSE: {data[1]}")

        print("===== NON-ZERO TEST DATA =====")
        paths = LoadData.getTempDirectories(args.data)
        non_zero_data = TrainFunctions.getTestNonZeroData(paths[2], args.batch_size)
        non_zero_data = TrainFunctions.EvaluateModel(model, non_zero_data)
        print(f"Test Loss: {non_zero_data[0]}")
        print(f"Test MSE: {non_zero_data[1]}")

    else:
        raise NotImplementedError("Training for pytorch is not implemented yet")
