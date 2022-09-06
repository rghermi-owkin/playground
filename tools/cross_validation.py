@hydra.main(version_base=None, config_path="../../conf/", config_name="config")
def main(params: DictConfig) -> None:

    # Get params

    # Load data
    dataset = load_tcga(
        cohort=cohort,
        features_histo=features_histo,
        normalization_exp=normalization_exp,
    )

    # Pre-process data
    data = preprocess_data(data)
    patient_ids = data["patient_ids"]

    # Cross-validation...
    for r in range(n_repeats):
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=r)
        for s, (train_patient_ids, valid_patient_ids) in enumerate(kf.split(patient_ids)):

            # Split data
            train_data, valid_data = {}, {}
            for key in data.keys():
                train_indices = np.isin(data[key]["patient_ids"], train_patient_ids)
                valid_indices = np.isin(data[key]["patient_ids"], valid_patient_ids)

                train_data[key]["patient_ids"] = data[key]["patient_ids"][train_indices]
                valid_data[key]["patient_ids"] = data[key]["patient_ids"][valid_indices]

                train_data[key]["X"] = data[key]["X"][train_indices]
                valid_data[key]["X"] = data[key]["X"][valid_indices]

            # Training...
            trainer = TorchTrainer()
            trainer.train(train_data, valid_data)

if __name__ == "__main__":
    main()
