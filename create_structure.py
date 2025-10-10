import os

# Define the directory structure
# Extended directory structure with Feedstock Characteristics Library and drop-in uncertainty (MTGP/physics-informed GP)
structure = {
    "OpenAD-lib": {
        "data": {
            "raw": {},
            "processed": {},
            "external": {}
        },
        "notebooks": {
            "ad_case_study.ipynb": "",
            "bayesian_doe.ipynb": "",
            "lstm_vs_ann.ipynb": "",
            "pinns_demo.ipynb": "",
            "feedstock_library_demo.ipynb": "",
            "mtgp_ad_integration.ipynb": ""
        },
        "src": {
            "bioprocess_lib": {
                "__init__.py": "",
                "models": {
                    "__init__.py": "",
                    "chemostat_haldene.py": "",
                    "reduced_ad_model.py": "",
                    "adm1_model.py": "",
                    "microbial_dynamics.py": ""
                },
                "optimisation": {
                    "__init__.py": "",
                    "scheduler.py": "",
                    "bayesian_doe.py": "",
                    "parameter_estimation.py": "",
                    "hyperparameter_tuning.py": ""
                },
                "ml": {
                    "__init__.py": "",
                    "gp_model.py": "",
                    "ann_model.py": "",
                    "lstm_model.py": "",
                    "pinns_model.py": "",
                    # new MTGP & physics-informed GP modules
                    "mtgp.py": "",
                    "physics_informed_gp.py": "",
                    "multi_fidelity.py": "",
                    "sparse_gp_utils.py": ""
                },
                "feedstock": {
                    "__init__.py": "",
                    # Core library: descriptors, distributions, sampling, ingestion for ADM1
                    "feedstock_library.py": "",            # main API: sample(), describe(), load_measurements()
                    "descriptors.py": "",                  # standard descriptors (TS, VS, C:N, BMP, carbohydrates/proteins/lipids, fibre fractions)
                    "distributions.py": "",                # fit/serialize distributions (parametric / non-parametric)
                    "measurement_utils.py": "",            # routines to clean/standardize lab measurements
                    "monte_carlo_sampler.py": "",          # create ensembles of ADM1 influent scenarios
                    "adm1_input_generator.py": "",         # map sampled descriptors -> ADM1 influent format
                    "literature_meta_analysis.md": ""      # notes / sources used for distributions
                },
                "ad_integration": {
                    "__init__.py": "",
                    "adm1_interface.py": "",               # run ADM1, set influent from feedstock library
                    "adm1_simulation_batch.py": "",        # run ensemble ADM1 MC runs and save outputs
                    "mtgp_ad_interface.py": ""             # pipeline glue: ADM1 outputs -> MTGP training/validation
                },
                "control": {
                    "__init__.py": "",
                    "mpc_controller.py": "",
                },
                "utils": {
                    "__init__.py": "",
                    "data_utils.py": "",
                    "visualisation.py": "",
                    "serialization.py": ""                 # save/load distributions, models, ensembles
                }
            },
            "tests": {
                "test_models.py": "",
                "test_optimisation.py": "",
                "test_ml.py": "",
                "test_control.py": "",
                # new tests
                "test_feedstock.py": "",
                "test_mtgp.py": "",
                "test_ad_integration.py": ""
            }
        },
        "scripts": {
            "run_ad_ensemble.py": "",
            "train_mtgp.py": "",
            "sample_feedstock.py": "",
            "deploy_example_mpc.py": ""
        },
        "examples": {
            "examples_adm1_input.json": "",
            "example_feedstock_distribution.json": "",
            "mtgp_trained_example.pkl": ""
        },
        "docs": {
            "index.md": "",
            "installation.md": "",
            "usage.md": "",
            "api_reference.md": "",
            "models_overview.md": "",
            # new documentation pages
            "feedstock_library.md": "",
            "mtgp_and_physics_informed_gps.md": "",
            "adm1_integration.md": ""
        },
        ".gitignore": "",
        "LICENSE": "",
        "README.md": "",
        "requirements.txt": "",
        "setup.py": "",
        "mkdocs.yml": ""
    }
}


# Function to create directories and files recursively
def create_structure(path, structure):
    for key, value in structure.items():
        full_path = os.path.join(path, key)
        if isinstance(value, dict):
            # Create directory
            os.makedirs(full_path, exist_ok=True)
            # Recursively create subdirectories/files
            create_structure(full_path, value)
        else:
            # Create file
            with open(full_path, 'w') as f:
                f.write(value)

# Create the root directory -  No need to create it again, the repo is the root
root_dir = "OpenAD-lib"
os.makedirs(root_dir, exist_ok=True)

# Populate the directory structure
create_structure(root_dir, structure)