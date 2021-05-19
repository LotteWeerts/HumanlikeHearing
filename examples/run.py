from absl import app
from flags import FLAGS
from humanlikehearing.dataset import DataSet
from humanlikehearing.asrsystem import ASRSystem
from humanlikehearing.testbattery import *

def main(argv):

    dataset = DataSet.create(FLAGS.dataset, FLAGS.data_path)
    asr_system = ASRSystem.create(FLAGS.asr_system_name, FLAGS.model_path)
  
    print('ASR system and datasets loaded.')
    asr_class_name = type(asr_system).__name__
    timestamp = int(time.time())
    folder = os.path.join(FLAGS.results_folder, 'test_report_{}_{}'.format(asr_class_name, timestamp))
    os.mkdir(folder)
    FLAGS.append_flags_into_file(os.path.join(folder, 'flags.txt'))
    print('Results stored in {}\n'.format(folder))

    # NORMALISATION TEST
    normalisation_test = NormalisationTest(
        folder, 
        sentences_per_condition=FLAGS.sentences_per_condition
    )
    normalisation_test.run(asr_system, dataset)

    # BANDPASS FILTER EXPERIMENT
    bandpass_test = BandpassTest(
        folder, 
        sentences_per_condition=FLAGS.sentences_per_condition,
        test_parameters={
            'bandpass_widths_semitones': [None] + list(range(60, 3, -1))
        })
    bandpass_test.run(asr_system, dataset)

    # CLIPPING EXPERIMENT.
    print('Initiating clipping test.')
    peak_clipping_test = PeakClippingTest(
        folder, 
        sentences_per_condition=FLAGS.sentences_per_condition
    )
    peak_clipping_test.run(asr_system, dataset)

    # MODULATION SENSITIVITY TEST. 
    modulation_sensitivity_test = ModulationSensitivityTest(
        folder, store_wavs=False, 
        sentences_per_condition=FLAGS.sentences_per_condition, 
        test_parameters={
            'prefilter': (7.75, None, 3.75, None),
            'spectral_filters': [(0.0, 0.25), (0.25, 0.75), (0.75, 1.75), (1.75, 3.75)],
            'temporal_filters': [(0.0, 0.25), (0.25, 0.75), (0.75, 1.75), (1.75, 3.75), (3.75, 7.75)],
            'combined_filters': [],
            'noise_types': ['white_noise'],
            'snr_levels_db': [10]
        }
    )
    modulation_sensitivity_test.run(asr_system, dataset)

    print('Initiating ModulationSensitivityTest notch test.')
    modulation_sensitivity_test = ModulationSensitivityTest(
        folder, store_wavs=False, 
        sentences_per_condition=FLAGS.sentences_per_condition, 
        test_parameters={
            'spectral_filters': [(0.0, 1.0), (1.0, 3.0), (3.0, 7.0), (7.0, 15), (15.0, None)], 
            'temporal_filters': [(0.0, 1.0), (1.0, 3.0), (3.0, 7.0), (7.0, 15), (15.0, None)],
            'combined_filters': [(None, None, None, None), (7.75, None, 3.75, None)],
            'snr_levels_db': [10],
            'noise_types': ['white_noise']
        }
    )
    modulation_sensitivity_test.run(asr_system, dataset)

    # TEMPORAL FINE STRUCTURE EXPERIMENT. 
    print('Initiating TFS per spectral region test.')
    temporal_fine_structure_test = TemporalFineStructureTest(
        folder, 
        sentences_per_condition=FLAGS.sentences_per_condition
    )
    temporal_fine_structure_test.run(asr_system, dataset, adaptive_srt_estimation_flag=False)

    # TARGET AND MASKER PERIODICITY EXPERIMENT. 
    periodicity_test = PeriodicityTest(
        folder, 
        store_wavs=False, 
        sentences_per_condition=FLAGS.sentences_per_condition
    )
    periodicity_test.run(asr_system, dataset)

    # COMPETING TALKER EXPERIMENT. 
    competing_talker_experiment = CompetingTalkerTest(
        folder, 
        sentences_per_condition=FLAGS.sentences_per_condition,
        test_parameters={
            'min_snr_db': -12,
            'max_snr_db': 36,
            'nr_talkers': [1, 2, 4, 8, 16],
            'target_filters': [None],
            'noise_filters': [None, 'modulate', 'noise_vocode_gammatone'],
        })
    competing_talker_experiment.run(asr_system, dataset, adaptive_srt_estimation_flag=False)

if __name__ == '__main__':
  app.run(main)