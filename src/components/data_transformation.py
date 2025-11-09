import sys
from dataclasses import dataclass
import os
from packaging import version


import numpy as np
import pandas as pd

import sklearn


from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


'''
NOTES:
1.normalize categorical text (strip extra spaces, collapse multiple spaces, lowercasing),
2.make the encoder robust to unseen categories (OneHotEncoder(handle_unknown='ignore')),
3.add FunctionTransformer cleaning inside the pipeline (and also normalize raw train/test just before transform to be safe),

# numeric pipeline
    This is easier

# categorical pipeline:
    #  - impute (most_frequent),
    #  - clean text (strip/lower/collapse spaces) via FunctionTransformer,
    #  - one-hot encode with ignore for unknown categories,
    #  - scale without centering (works with sparse)    

TO KNOW:    
OneHotEncoder(handle_unknown='ignore') prevents the ValueError you saw when the test set has categories
unseen in training.

The cleaning step removes trailing spaces and normalizes case (so 'Yamaha Fazer ' becomes 
'yamaha fazer') — this usually eliminates many "unknown category" problems caused by inconsistent 
formatting.

I added verbose_feature_names_out=False to ColumnTransformer to keep consistent feature naming 
behavior; remove if your sklearn version has different behavior.

If your dataset still has many rare/unique categories and you prefer to map them to an "other" token, 
we can add a RareCategoryEncoder step (or map unknowns after fitting).

'''


def _clean_text_array(X):
    """
    Expects a 2D array-like or pandas DataFrame/Series.
    Collapses multiple spaces, strips leading/trailing spaces and lowercases strings.
    Returns numpy array with cleaned strings.
    """
    # If X is a DataFrame or 2D array, apply cleaning column-wise
    if hasattr(X, "astype") and hasattr(X, "apply"):
        # pandas DataFrame/Series
        return X.astype(str).apply(lambda col: col.str.replace(r"\s+", " ", regex=True).str.strip().str.lower())
    else:
        # numpy array -> convert to DataFrame for convenience
        df = pd.DataFrame(X)
        return df.astype(str).apply(lambda col: col.str.replace(r"\s+", " ", regex=True).str.strip().str.lower())


@dataclass
class DataTransformationConfig:
    # fixed filename typo: preprocessor (not proprocessor)
    preprocessor_obj_file_path: str = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    # def get_data_transformation_object(self):
    #     """
    #     Create and return a ColumnTransformer that:
    #      - imputes and scales numerical columns
    #      - imputes, cleans, encodes categorical columns and scales (no mean centering for sparse)
    #     """
    #     try:
    #         numerical_columns = ["Year", "Present_Price", "Kms_Driven", "Owner"]
    #         categorical_columns = ["Car_Name", "Fuel_Type", "Seller_Type", "Transmission"]

    #         # numeric pipeline
    #         num_pipeline = Pipeline(
    #             steps=[
    #                 ("imputer", SimpleImputer(strategy="median")),
    #                 ("scaler", StandardScaler()),
    #             ]
    #         )

    #         # categorical pipeline:
    #         #  - impute (most_frequent),
    #         #  - clean text (strip/lower/collapse spaces) via FunctionTransformer,
    #         #  - one-hot encode with ignore for unknown categories,
    #         #  - scale without centering (works with sparse)
    #         cat_pipeline = Pipeline(
    #             steps=[
    #                 ("imputer", SimpleImputer(strategy="most_frequent")),
    #                 ("text_cleaner", FunctionTransformer(lambda X: _clean_text_array(X), validate=False)),
    #                 ("one_hot", OneHotEncoder(handle_unknown="ignore", sparse=False)),
    #                 ("scaler", StandardScaler(with_mean=False)),
    #             ]
    #         )

    #         logging.info(f"Categorical columns: {categorical_columns}")
    #         logging.info(f"Numerical columns: {numerical_columns}")

    #         preprocessor = ColumnTransformer(
    #             transformers=[
    #                 ("num_pipeline", num_pipeline, numerical_columns),
    #                 ("cat_pipeline", cat_pipeline, categorical_columns),
    #             ],
    #             remainder="drop",
    #             verbose_feature_names_out=False,
    #         )

    #         return preprocessor

    #     except Exception as e:
    #         raise CustomException(e, sys)

    def get_data_transformation_object(self):
        try:
            numerical_columns = ["Year", "Present_Price", "Kms_Driven", "Owner"]
            categorical_columns = ["Car_Name", "Fuel_Type", "Seller_Type", "Transmission"]

            num_pipeline = Pipeline(
                steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ]
            )

            # Choose correct OneHotEncoder argument depending on sklearn version
            if version.parse(sklearn.__version__) >= version.parse("1.2"):
                ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            else:
                ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    # ("text_cleaner", FunctionTransformer(lambda X: _clean_text_array(X), validate=False)),
                    ("text_cleaner", FunctionTransformer(_clean_text_array, validate=False)),
                    ("one_hot", ohe),
                    ("scaler", StandardScaler(with_mean=False)),
                ]
            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns),
                ],
                remainder="drop",
                verbose_feature_names_out=False,
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)
    
    def initiate_data_transformation(self, train_path: str, test_path: str):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read Train and Test data into DataFrames.")

            preprocessor_obj = self.get_data_transformation_object()

            target_column_name = "Selling_Price"
            numerical_columns = ["Year", "Present_Price", "Kms_Driven", "Owner"]
            categorical_columns = ["Car_Name", "Fuel_Type", "Seller_Type", "Transmission"]

            # Separate input features and target
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Cleaning categorical text columns (strip/lower/collapse spaces) before transform.")

            # Clean categorical columns in-place (ensures same cleaning for train & test)
            for col in categorical_columns:
                if col in input_feature_train_df.columns:
                    input_feature_train_df[col] = (
                        input_feature_train_df[col].astype(str)
                        .str.replace(r"\s+", " ", regex=True)
                        .str.strip()
                        .str.lower()
                    )
                if col in input_feature_test_df.columns:
                    input_feature_test_df[col] = (
                        input_feature_test_df[col].astype(str)
                        .str.replace(r"\s+", " ", regex=True)
                        .str.strip()
                        .str.lower()
                    )

            logging.info("Applying preprocessing object on training dataframe and testing dataframe.")

            # Fit on train, transform both
            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)

            # Combine features and target
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info("Saved preprocessing object.")

            # Persist preprocessor
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj,
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e, sys)
