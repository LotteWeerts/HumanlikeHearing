from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_string('dataset', 'TestDataSet', 'Name of DataSet class that should be loaded.')
flags.DEFINE_string('data_path', './examples/test_data', 'Path to data set folder.')
flags.DEFINE_string('noise_path', '', 'Path to noise folder.')
flags.DEFINE_string('asr_system_name', 'TestASR', 'Name of ASR system class that should be loaded.')
flags.DEFINE_string('results_folder', './results', 'Location where results should be stored.')
flags.DEFINE_string('model_path', '', 'Model file path for ASR system.')
flags.DEFINE_integer('sentences_per_condition', 100, 'Number of sentences from the dataset that should be tested per condition in test.')