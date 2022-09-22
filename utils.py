import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import os
import functools


def aggregate_dataset(df, grouping_field_list, array_field):
    df = df.groupby(grouping_field_list)['encounter_id',
                                         array_field].apply(
        lambda x: x[array_field].values.tolist()).reset_index().rename(columns={
        0: array_field + "_array"})

    dummy_df = pd.get_dummies(df[array_field + '_array'].apply(pd.Series).stack()).sum(level=0)
    dummy_col_list = [x.replace(" ", "_") for x in list(dummy_df.columns)]
    mapping_name_dict = dict(zip([x for x in list(dummy_df.columns)], dummy_col_list))
    concat_df = pd.concat([df, dummy_df], axis=1)
    new_col_list = [x.replace(" ", "_") for x in list(concat_df.columns)]
    concat_df.columns = new_col_list

    return concat_df, dummy_col_list


def cast_df(df, col, d_type=str):
    return df[col].astype(d_type)


def impute_df(df, col, impute_value=0):
    return df[col].fillna(impute_value)


def preprocess_df(df, categorical_col_list, numerical_col_list, predictor, categorical_impute_value='nan',
                  numerical_impute_value=0):
    df[predictor] = df[predictor].astype(float)
    for c in categorical_col_list:
        df[c] = cast_df(df, c, d_type=str)
    for numerical_column in numerical_col_list:
        df[numerical_column] = impute_df(df, numerical_column, numerical_impute_value)
    return df


# adapted from https://www.tensorflow.org/tutorials/structured_data/feature_columns
def df_to_dataset(df, predictor, batch_size=32):
    df = df.copy()
    labels = df.pop(predictor)
    ds = tf.data.Dataset.from_tensor_slices((dict(df), labels))
    ds = ds.shuffle(buffer_size=len(df))
    ds = ds.batch(batch_size)
    return ds


# build vocab for categorical features
def write_vocabulary_file(vocab_list, field_name, default_value, vocab_dir='./survival_vocab/'):
    output_file_path = os.path.join(vocab_dir, str(field_name) + "_vocab.txt")
    # put default value in first row as TF requires
    vocab_list = np.insert(vocab_list, 0, default_value, axis=0)
    df = pd.DataFrame(vocab_list).to_csv(output_file_path, index=None, header=None)
    return output_file_path


def build_vocab_files(df, categorical_column_list, default_value='00'):
    vocab_files_list = []
    for c in categorical_column_list:
        v_file = write_vocabulary_file(df[c].unique(), c, default_value)
        vocab_files_list.append(v_file)
    return vocab_files_list


def show_group_stats_viz(df, group):
    print(df.groupby(group).size())
    print(df.groupby(group).size().plot(kind='barh'))


'''
Adapted from Tensorflow Probability Regression tutorial  https://github.com/tensorflow/probability/blob/master/tensorflow_probability/examples/jupyter_notebooks/Probabilistic_Layers_Regression.ipynb
'''


def posterior_mean_field(kernel_size, bias_size=0, dtype=None):
    n = kernel_size + bias_size
    c = np.log(np.expm1(1.))
    return tf.keras.Sequential([
        tfp.layers.VariableLayer(2 * n, dtype=dtype),
        tfp.layers.DistributionLambda(lambda t: tfp.distributions.Independent(
            tfp.distributions.Normal(loc=t[..., :n],
                                     scale=1e-5 + tf.nn.softplus(c + t[..., n:])),
            reinterpreted_batch_ndims=1)),
    ])


def prior_trainable(kernel_size, bias_size=0, dtype=None):
    n = kernel_size + bias_size
    return tf.keras.Sequential([
        tfp.layers.VariableLayer(n, dtype=dtype),
        tfp.layers.DistributionLambda(lambda t: tfp.distributions.Independent(
            tfp.distributions.Normal(loc=t, scale=1),
            reinterpreted_batch_ndims=1)),
    ])


def demo(feature_column, example_batch):
    feature_layer = tf.keras.layers.DenseFeatures(feature_column)
    print(feature_layer(example_batch))
    return feature_layer(example_batch)


def calculate_stats_from_train_data(df, col):
    mean = df[col].describe()['mean']
    std = df[col].describe()['std']
    return mean, std


def create_tf_numerical_feature_cols(numerical_col_list, train_df):
    tf_numeric_col_list = []
    for c in numerical_col_list:
        mean, std = calculate_stats_from_train_data(train_df, c)
        tf_numeric_feature = create_tf_numeric_feature(c, mean, std)
        tf_numeric_col_list.append(tf_numeric_feature)
    return tf_numeric_col_list

def reduce_dimension_ndc(df, ndc_df):
    '''
    df: pandas dataframe, input dataset
    ndc_df: pandas dataframe, drug code dataset used for mapping in generic names
    return:
        df: pandas dataframe, output dataframe with joined generic drug name
    '''

    df['generic_drug_name'] = ndc_df['Non-proprietary Name']
    return df


def select_first_encounter(df):
    '''
    df: pandas dataframe, dataframe with all encounters
    return:
        - first_encounter_df: pandas dataframe, dataframe with only the first encounter for a given patient
    '''

    first_encounter_df = df.sort_values(['encounter_id'], ascending=True).groupby('patient_nbr').head(1)

    return first_encounter_df


def patient_dataset_splitter(df, patient_key='patient_TrustNumber'):
    '''
    df: pandas dataframe, input dataset that will be split
    patient_key: string, column that is the patient id

    return:
     - train: pandas dataframe,
     - validation: pandas dataframe,
     - test: pandas dataframe,
    '''

    df.iloc[np.random.permutation(len(df))]
    unique_values = df[patient_key].unique()
    total_values = len(unique_values)
    sample_size = round(total_values * 0.6)
    val_size = round(total_values * 0.2)
    train = df[df[patient_key].isin(unique_values[:sample_size])].reset_index(drop=True)
    validation = df[df[patient_key].isin(unique_values[:val_size])].reset_index(drop=True)
    val_test = df[df[patient_key].isin(unique_values[sample_size:])].reset_index(drop=True)
    test = val_test.drop(validation.index)
    
        
    return train, validation, test


def create_tf_categorical_feature_cols(categorical_col_list,
                                       vocab_dir='./survival_vocab/'):
    '''
    categorical_col_list: list, categorical field list that will be transformed with TF feature column
    vocab_dir: string, the path where the vocabulary text files are located
    return:
        output_tf_list: list of TF feature columns
    '''
    output_tf_list = []
    for c in categorical_col_list:
        vocab_file_path = os.path.join(vocab_dir, c + "_vocab.txt")
        '''
        Which TF function allows you to read from a text file and create a categorical feature
        You can use a pattern like this below...
        tf_categorical_feature_column = tf.feature_column.......

        '''
        tf_categorical_feature_column = tf.feature_column.categorical_column_with_vocabulary_file(
            key=c, vocabulary_file=vocab_file_path, num_oov_buckets=0)
        tf_categorical_feature_column = tf.feature_column.embedding_column(tf_categorical_feature_column, dimension=10)
        output_tf_list.append(tf_categorical_feature_column)
    return output_tf_list


def normalize_numeric_with_zscore(col, mean, std):
    '''
    This function can be used in conjunction with the tf feature column for normalization
    '''
    return (col - mean) / std


def create_tf_numeric_feature(col, MEAN, STD, default_value=0):
    '''
    col: string, input numerical column name
    MEAN: the mean for the column in the training data
    STD: the standard deviation for the column in the training data
    default_value: the value that will be used for imputing the field

    return:
        tf_numeric_feature: tf feature column representation of the input field
    '''
    normalizer = functools.partial(normalize_numeric_with_zscore, mean=MEAN, std=STD)
    tf_numeric_feature = tf.feature_column.numeric_column(
        key=col, default_value=default_value, normalizer_fn=normalizer, dtype=tf.float64)
    return tf_numeric_feature


def get_mean_std_from_preds(diabetes_yhat):
    '''
    diabetes_yhat: TF Probability prediction object
    '''
    m = diabetes_yhat.mean()
    s = diabetes_yhat.stddev()
    return m, s


def get_student_binary_prediction(df, col):
    '''
    df: pandas dataframe prediction output dataframe
    col: str,  probability mean prediction field
    return:
        student_binary_prediction: pandas dataframe converting input to flattened numpy array and binary labels
    '''

    student_binary_prediction = df[col].apply(lambda x: 1 if x >= 1 else 0).values
    print(f'### Transformed to numpy: {type(student_binary_prediction)}, shape: {student_binary_prediction.shape}')
    return student_binary_prediction

def patient_dataset_splitter_compare(df, patient_key='patient_nbr'):
    '''
    df: pandas dataframe, input dataset that will be split
    patient_key: string, column that is the patient id
    return:
     - train: pandas dataframe,
     - validation: pandas dataframe,
     - test: pandas dataframe,
    '''
    
    df.iloc[np.random.permutation(len(df))]
    unique_values = df[patient_key].unique()
    total_values = len(unique_values)
    sample_size = round(total_values * 0.6)
    test_size = round(total_values * 0.4)
    train = df[df[patient_key].isin(unique_values[:sample_size])].reset_index(drop=True)
    test = df[df[patient_key].isin(unique_values[:test_size])].reset_index(drop=True)
    
    p_inds = train[train['Event']==1].index.tolist()
    np_inds = train[train['Event']==0].index.tolist()
    np_sample = sample(np_inds, 12*len(p_inds))
    train = train.loc[p_inds + np_sample]
    train['Event'].sum()/len(train)
    
    p_inds = test[test['Event']==1].index.tolist()
    np_inds = test[test['Event']==0].index.tolist()
    np_sample = sample(np_inds, 12*len(p_inds))
    test = test.loc[p_inds + np_sample]
    test['Event'].sum()/len(test)
    
    return train, test

def patient_dataset_splitter_balance(df, patient_key='patient_TrustNumber'):
    '''
    df: pandas dataframe, input dataset that will be split
    patient_key: string, column that is the patient id

    return:
     - train: pandas dataframe,
     - validation: pandas dataframe,
     - test: pandas dataframe,
    '''
    df.iloc[np.random.permutation(len(df))]
    unique_values = df[patient_key].unique()
    total_values = len(unique_values)
    sample_size = round(total_values * 0.6)
    val_size = round(total_values * 0.2)
    train = df[df[patient_key].isin(unique_values[:sample_size])].reset_index(drop=True)
    validation = df[df[patient_key].isin(unique_values[:val_size])].reset_index(drop=True)
    val_test = df[df[patient_key].isin(unique_values[sample_size:])].reset_index(drop=True)
    test = val_test.drop(validation.index)
    
    p_inds = train[train['Event']==1].index.tolist()
    np_inds = train[train['Event']==0].index.tolist()
    np_sample = sample(np_inds, 12*len(p_inds))
    train = train.loc[p_inds + np_sample]
    train['Event'].sum()/len(train)
    
    p_inds = validation[validation['Event']==1].index.tolist()
    np_inds = validation[validation['Event']==0].index.tolist()
    np_sample = sample(np_inds, 12*len(p_inds))
    validation = validation.loc[p_inds + np_sample]
    validation['Event'].sum()/len(validation)
    
    p_inds = test[test['Event']==1].index.tolist()
    np_inds = test[test['Event']==0].index.tolist()
    np_sample = sample(np_inds, 12*len(p_inds))
    test = test.loc[p_inds + np_sample]
    test['Event'].sum()/len(test)
    
        
    return train, validation, test
