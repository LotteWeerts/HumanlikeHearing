from .library import speech_voltmeter_svp56 as svp56
from .library import a_weighting

import numpy as np
import scipy
import soundfile
import librosa

class Sound(np.ndarray):
    """
    A Sound object behaves as a numpy ndarray but incorporates level_dB and samplerate_Hz
    properties. The class also implements several functions for mixing different noises
    and resampling sounds.

    A Sound object can be read either from a file or can be created directly from a data
    array. This data is expected to use a floating point format which usually ranges in
    [-1., 1.]. More specifically, in computing the sound levels it is assumed the data is 
    given in Pascals.
    """

    def __new__(cls, input_sound, samplerate_Hz=None, level_dB=None):
        """A subclass of numpy's np.ndarray needs to use __new__ rather than __init__.
        See here for more info: https://numpy.org/doc/stable/user/basics.subclassing.html
        """
        if isinstance(input_sound, str):
            input_sound, original_samplerate_Hz = soundfile.read(input_sound)
            initiated_from_file = True            
        else:
            initiated_from_file = False
        # Cast ndarray instance to Sound class type.
        obj = np.asarray(input_sound).view(cls)
        if initiated_from_file: # Initiate with samplerate from the sound file.
            obj.samplerate_Hz = original_samplerate_Hz
            if samplerate_Hz is not None and samplerate_Hz != original_samplerate_Hz:
                obj = cls.resample(obj, samplerate_Hz)
        elif samplerate_Hz is not None:
            obj.samplerate_Hz = samplerate_Hz
        else:
            raise Exception('samplerate_Hz needs to be defined if Sound is initiated from array-like input.')
        if level_dB is not None:
            obj.level_dB = level_dB
        return obj

    def __array_wrap__(self, obj, context=None):
        x = np.ndarray.__array_wrap__(self, obj, context)
        if not hasattr(x, 'samplerate_Hz') and hasattr(self, 'samplerate_Hz'):
            x.samplerate_Hz = self.samplerate_Hz
        if context is not None:
            ufunc = context[0]
            args = context[1]
        return x

    def __array_finalize__(self, obj):
        if obj is None: return

    @property
    def level_dB(self):
        """The sound level (rms) in dB, assuming the data is in Pascals.
        
        Uses the simple rms to compute the sound level, for more advanced sound
        level estimates see get_advanced_level().
        """
        return self.get_advanced_level(False, False)

    @level_dB.setter
    def level_dB(self, new_level_dB):
        """Adjusts level of the sound to a given new_level (dB).
        
        Uses the simple rms to compute the sound level, for more intricate sound
        level estimates see set_advanced_level().
        """
        self.set_advanced_level(new_level_dB , False, False)

    @property
    def samplerate_Hz(self):
        return self._samplerate_Hz

    @samplerate_Hz.setter
    def samplerate_Hz(self, new_samplerate_Hz):
        if hasattr(self, '_samplerate_Hz'):
            raise AttributeError('Cannot reassign samplerate_Hz directly, please use resample() instead.')
        self._samplerate_Hz = new_samplerate_Hz

    def write(self, path):
        """Write sound to path."""
        soundfile.write(path, self, self.samplerate_Hz)

    def get_advanced_level(self, a_weighting=False, speech_volt_meter=False):
        """Compute sound level taking either a_weighting and/or speech activity into account."""
        sound = self
        if a_weighting:
            sound = Sound.apply_a_weighting(sound)
        if speech_volt_meter:
            return Sound.compute_speech_voltmeter_level(sound)
        return Sound.compute_rms_level_as_pascal(sound)

    def set_advanced_level(self, new_level_dB, a_weighting=False, speech_volt_meter=False):
        """Set level of sound to new level taking a weighting and/or speech activity into account."""
        current_level = self.get_advanced_level(a_weighting, speech_volt_meter)
        self *= Sound.gain(current_level, new_level_dB)

    @classmethod
    def apply_a_weighting(cls, sound):
        """Applies a_weighting to a sound."""
        b, a = a_weighting.a_weighting(sound.samplerate_Hz)
        return Sound(scipy.signal.lfilter(b, a, sound), sound.samplerate_Hz)

    @classmethod
    def compute_rms_level_as_pascal(cls, sound):
        """Computes the sound level assuming the data is given in Pascals."""
        rms_value = np.sqrt(np.mean((sound-np.mean(sound))**2))
        rms_dB = 20.0*np.log10(rms_value/2e-5)
        return rms_dB

    @classmethod
    def compute_speech_voltmeter_level(cls, sound, frame_duration_ms=10):
        """Computes sound level adjusted such that silences in speech are ignored.
        Uses active voice detection following the p56 recommendation. Assumes
        data is given in Pascals.
        """

        state = svp56.SVP56_state()
        svp56.init_speech_voltmeter(state, sound.samplerate_Hz)
        original_level = sound.level_dB
        sound_scaled = Sound(sound, sound.samplerate_Hz)

        # The voice activity detector only works well for a -20 to 0 dB range.
        sound_scaled.level_dB = 0
        # Adjust scale against the reference of 20 microPascal.
        sound_scaled = (sound_scaled - np.mean(sound_scaled))/2e-5
        true_level_at_0_db = svp56.speech_voltmeter(sound_scaled, state)
        return true_level_at_0_db + original_level

    @classmethod
    def gain(cls, old_level, new_level):
        """Computes the gain needed adjust a sound level from old_level to new_level."""
        return 10**((new_level-old_level)/20.)

    @classmethod
    def resample(cls, sound, new_samplerate_Hz):
        """Returns a resampled version of the given sound with the target samplerate."""
        if new_samplerate_Hz != sound.samplerate_Hz:
            resampled = librosa.resample(sound, sound.samplerate_Hz, new_samplerate_Hz)
            return Sound(resampled, samplerate_Hz=new_samplerate_Hz)
        return sound