# ML_project

The main part of the code concerning the hacking model and the shadow model is in the file "hack_class.py." In this file, there are three classes: HackingModels, ShadowModels, and ShadowModel. These three classes are used for all the following experiments.

To replicate the graph, execute the code in the "plot_result.ipynb" notebook. This notebook uses data that has already been computed and stored in the "data" folders, saved in JSON files across all folders, to create the graph used in the report.

To replicate the experiments done for this project, it is possible to execute all the notebooks with names starting with "local." These Jupyter notebooks enable the regeneration of data and the storage of data in some JSON files.
- adult\local_adult_result : Fig 4
- MNIST\local_MNIST_result : Fig 2
- MNIST\local_test_model : Fig 3
- MNIST\local_shadow : Fig 5
- overfitting\local_of_logistic : Fig 6
- Data sythesis\local_dataset_generation : Fig 7 et 8
- diffrencial_privacy\local_privacy : Fig 9

All the files starting with "launch" can be used to regenerate the data, but contrary to the files starting with "local," the execution is performed via SSH on the Polytechnique computers.