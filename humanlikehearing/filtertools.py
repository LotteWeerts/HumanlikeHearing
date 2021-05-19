import numpy as np
import parselmouth
import scipy
from .sound import Sound

"""
filtertools.py contains helper functions for filtering. This mainly involves:
* Functions that generate channel spacing (erb or greenwood spacing)
* Functions that can apply specific filters (e.g. butterworth or elliptic filters)
  while returning Sound objects
* A vocoder that implements vocoding required by several tests in the battery.

Thanks to Michal Stone for kindly sharing his code for erb FIR filter spacing, which 
was written in 2006 while at the Auditory Perception Group at University of Cambridge 
(group head Prof. Brian C.J. Moore). 

DISCLAIMER: this code, particularly the vocoder code, has been written for
the specific experiments implemented in the HumanlikeHearing test battery. 
Extending the vocoder code to other experiments, particularly chosing the correct 
parameters, is non trivial. This code comes with absolutely no warranty. 
"""

def tone(frequency_hertz, samplerate_hertz, duration_sec=None, frames=None):
    if frames is None and duration_sec is not None:
        frames = int(duration_sec*samplerate_hertz)
    elif duration_sec is None and frames is not None:
        duration_sec = frames/samplerate_hertz
    elif duration_sec is None and frames is None:
        raise ValueError('duration_sec and frames cannot both be None.')
    else:
        raise ValueError('frames and duration_sec cannot both be defined.')
    ts = np.linspace(0, duration_sec, frames)
    return np.sin(2*np.pi*frequency_hertz*ts)

def erb_space(n, f_stop, f_start=100):
    """
    # n bands, fstop is upper limit of speech info frequency
    Default to 100 Hz for lowest edge of speech start (100 Hz was lowest defined ERB in fig 3.10 of Intro to Psych ed 4).
    """

    def hertz_to_erb_number(freq):
        return 21.4*np.log10(1 + 4.37*freq/1000)

    def erb_number_to_hertz(erb_number):
        return 1000*(10**(erb_number/21.4) - 1)/4.37

    e_stop = hertz_to_erb_number(f_stop)
    e_start = hertz_to_erb_number(f_start)
    e_edges = np.linspace(e_start, e_stop, n + 1)
    edges = erb_number_to_hertz(e_edges)
    return edges

def greenwood_space(low, high, N):
    """Generates N filter boundaries following Greenwood spacing between low and high (in Hz)

    Returns list of N filter boundaries, where each boundary is a tuple of a lower and
    upper frequency in Hz.
    """

    def mm_to_frequency(mm):
        a = .06  # appropriate for measuring basilar membrane length in mm
        k = 165.4
        return 165.4 * (10**(a * mm)- 1)

    def frequency_to_mm(freq):
        a = .06  # appropriate for measuring basilar membrane length in mm
        k = 165.4
        return (1/a) * np.log10(freq/k + 1)

    # Set up equally spaced places on the basilar membrane
    places = np.arange(N + 1)*(frequency_to_mm(high) - frequency_to_mm(low))/N + frequency_to_mm(low)
    # convert these back to frequencies
    freqs = mm_to_frequency(places)
    #centres = np.zeros(N)
    #centres = (places[:N] + places[1:N+1])/2
    #center = mm_to_frequency(centres)
    lower = freqs[:N]
    upper = freqs[1:]

    return list(zip(lower, upper))

def elliptic_filter(sound, order, rp, rs, low=None, high=None):
    if low is None and high is None:
        raise ValueError('Either high or low must be defined for elliptic filter.')
    elif low is None:
        btype = 'highpass'
        Wn = [high]
    elif high is None:
        btype = 'lowpass'
        Wn = [low]
    else:
        btype = 'bandpass'
        Wn = np.array([low, high])
    sos = scipy.signal.ellip(order, rp, rs, Wn, btype=btype, fs=sound.samplerate_Hz, output='sos')
    filtered = scipy.signal.sosfilt(sos, sound)
    return Sound(filtered, sound.samplerate_Hz)

def butter_filter(sound, order, low=None, high=None, zero_pole=True):
    if low is None and high is None:
        raise ValueError('Either high or low must be defined for butterworth filter.')
    elif low is None:
        btype = 'highpass'
        Wn = [high]
    elif high is None:
        btype = 'lowpass'
        Wn = [low]
    else:
        btype = 'bandpass'
        Wn = np.array([low, high])
        
    if zero_pole:
        order = int(order)
        if order % 2 != 0:
            raise ValueError('An even order is required for zero phase shift filters.')
        sos = scipy.signal.butter(order//2, Wn, btype=btype, fs=sound.samplerate_Hz, output='sos')
        filtered = scipy.signal.sosfiltfilt(sos, sound)
    else:
        sos = scipy.signal.butter(order, Wn, btype=btype, fs=sound.samplerate_Hz, output='sos')
        filtered = scipy.signal.sosfilt(sos, sound)

    sos = scipy.signal.butter(order, Wn, btype=btype, fs=sound.samplerate_Hz, output='sos')
    filtered = scipy.signal.sosfilt(sos, sound)
    return Sound(filtered, sound.samplerate_Hz)

def fir_filter(sound, bandpass_filter):
    channel_signal = scipy.signal.lfilter(bandpass_filter, 1, sound)
    # Compensating shift to time-align all filter outputs.
    dly_shift = len(bandpass_filter)//2 
    proc_len = len(sound)
    valid_len = proc_len - dly_shift # _advance_ filter outputs.
    channel_signal[:valid_len] = channel_signal[dly_shift:proc_len + 1] # time advance
    channel_signal[valid_len:proc_len] = 0 # Kill the rest.
    return Sound(channel_signal, sound.samplerate_Hz)

def atleast_1d_tuple(arr):
    if isinstance(arr, str):
        return tuple([arr])
    try:
        return tuple(arr)
    except TypeError:
        return tuple([arr])

def apply_filter(sound, filterbank_type, filter_params):
    filterbank_type = atleast_1d_tuple(filterbank_type)
    if filterbank_type[0] == 'butter':
        filtered_sound = butter_filter(sound, filterbank_type[1], *filter_params)
    elif filterbank_type[0] == 'elliptic':
        filtered_sound = elliptic_filter(
            sound, 
            filterbank_type[1], 
            filterbank_type[2], 
            filterbank_type[3], 
            *filter_params
        )
    elif filterbank_type[0] == 'fir':
        filtered_sound = fir_filter(sound, filter_params[0])
    else:
        raise ValueError('Unknown filterbank type: {}'.format(filterbank_type))
    return filtered_sound

def apply_envelope_extraction(sound, envelope_method):
    envelope_method = atleast_1d_tuple(envelope_method)
    if envelope_method[0] == 'butter':
        envelope = butter_filter(
            np.abs(sound), 
            order=envelope_method[1],
            low=envelope_method[2]
        )
    elif envelope_method[0] == 'hilbert':
        envelope = Sound(np.abs(scipy.signal.hilbert(sound)), sound.samplerate_Hz)
    else:
        raise ValueError('Unknown envelope method: {}'.format(envelope_method))
    return envelope

def generate_f0_pulses(sound, interpolate=True):
    parselsound = parselmouth.Sound(sound, sound.samplerate_Hz)
    manipulation = parselmouth.praat.call(parselsound, "To Manipulation", 0.01, 75, 600)
    pitch_tier = parselmouth.praat.call(manipulation, "Extract pitch tier")

    pitch = parselsound.to_pitch(time_step=0.01)
    f0_contours = pitch.selected_array['frequency']
    time_in_second = pitch.xs()
    parselmouth.praat.call(pitch_tier, "Remove points between", 0, parselsound.duration)

    if interpolate:
        zeros = (f0_contours == 0)
        mean_frequency = np.median(f0_contours[~zeros])
        f0_contours[0], zeros[0] = mean_frequency, False
        f0_contours[-1], zeros[-1] = mean_frequency, False
        interpolator = scipy.interpolate.PchipInterpolator(time_in_second[~zeros], np.log10(f0_contours[~zeros]))
        f0_contours = 10**interpolator(time_in_second)
    
    for i, t in enumerate(time_in_second):
        parselmouth.praat.call(pitch_tier, "Add point", t, f0_contours[i])

    point_process = parselmouth.praat.call(pitch_tier, "To PointProcess")
    pulse_train = parselmouth.praat.call(
        point_process, 
        "To Sound (phonation)",
        sound.samplerate_Hz,
        1.0, 0.05, 0.7, 0.03, 3.0, 4.0
    )
    pulse_train = np.squeeze(pulse_train)
    new_sound = Sound(pulse_train, sound.samplerate_Hz)
    return new_sound, f0_contours, time_in_second

def erb_fir_filters(signal, edges, fs): 
    """
    Generates a bank of FIR filters for a given set of erb edges following the approach of 
    Hopkins et al. (2010), such that each filter has a response of -6 dB (relative to the
    peak response) at the frequencies at which its response intersects with the 
    response of the two neighbouring filters. This code was adapted from Matlab code that 
    was kindly shared by Micheal Stone, which was written in 2006 while at the Auditory 
    Perception Group at University of Cambridge (group head Prof. Brian C.J. Moore). 
    
    CAVEAT EMPTOR: this code was written for this specfic TFS experiment only, no guarantees
    can be made for its correctness outside of the boundaries of this specific experiment. 

    To generate bandpass filters for analysis/synthesis, need some magic numbers to define
    transition widths/time domain impulse response. These are chosen so that for nchans=16, 
    filters have similar or wider freq response to ERB, so that no long time impulse 
    response filters generated.  Also set minimum number of taps to ensure that tails fall 
    to < -65 dB. Also adjust transition width so that for low number of channels, where 
    spacing is high, it still chooses moderately steep filters to get good channel separation
    Two attempts are made to tame the tails: 
    * (a) in lpf design use kaiser with beta=6, 
    * (b) then after bpf has been generated by convolution of lpf with hpf, use beta=3.
    """

    firlen0 = int(2*np.floor(2.5*fs/1000)) # 80 for 16k, minimum number of taps, MUST be EVEN (pref divisible by 4).
    max_mult = 8                      # Magic number of 8, controls maximum fir size likely for channel splits.
    qnorm = 3.3                       # Magic number of about 3.3.
    ramp_res = 512                    # Resolution of ramp to fade out tails of tilted synthesis filters.
    # Ramp to attenuate lf tails of synthesis filter.
    ramp = 0.5*(1.2 - 0.8*np.cos(0.5*np.pi*np.arange(ramp_res)/ramp_res))  

    nchans = len(edges) - 1        # Number of processing channels.
    bp_lf = edges[:nchans]         # Lower corner frequencies for channel splits.
    bp_hf = edges[1:nchans+1]      # Upper corner frequencies for channel splits.
    bp_cf = 0.5*(bp_hf + bp_lf)    # Centre frequencies/crossovers.
    bp_bw = bp_hf - bp_lf          # Bandwidth
    bp_norm = np.sqrt(bp_bw)       # Signal _gain_ per filter when filtering WHITE noise.
    bp_norm = bp_norm[0]/bp_norm   # Invert and make zero dB at DC.

    # Shape of transition is dependent on difference between the adjacent cfs.
    delta_cf = bp_cf[1:nchans] - bp_cf[:nchans-1]

    # Measure of steepness, is 'q' around transition.
    q = 0.5*(bp_cf[1:nchans] + bp_cf[:nchans-1])/(bp_cf[1:nchans] - bp_cf[:nchans-1])

    # Effectively broaden transitions for high nchans, tighten for low
    # nchans & reduces number of taps for low freq channels.
    delta_cf = delta_cf*(q/qnorm)
    
    # Preallocate bpfs. Note: extra +1 is added for
    # a) Length of filter
    # b) Centre tap (i.e. odd order filters).
    #size = max_mult*(int(100*firlen0/delta_cf[0]) + int(100*firlen0/delta_cf[1])) + 2
    #bpf = np.zeros((size, nchans))

    bpf = []
    
    # Channel envelope lpf is fixed for all channels, fc approx 1/2 octave above 32 Hz.
    # No need to worry about phase since using bidirectional filters.
    channel_lpfB, channel_lpfA = scipy.signal.ellip(3,.1,35,45/(fs/2))
    
    # Generate analysis filterbank.
    for index in range(nchans): 
        # Design filters, gradually increase window to reduce tails.
        # Filter is designed in two stages, depending on adjacent channels,
        # high pass first then low pass.
    
        if index != 0:
            # Index of middle position of lpf.
            mid_lpf = int(len(lpf_chan)/2) 
            # High-pass is complementary of low-pass from previous channel.
            hpf_chan = -lpf_chan
            hpf_chan[mid_lpf] += 1
        else:
            # Special case at the start.
            hpf_chan = np.atleast_1d(1)

        # firlen adapts to transition width, last channel is irrelevant. 
        if index < nchans - 1:
            # Must ALWAYS end up EVEN, so fir1() turns it ODD.
            firlen = max(firlen0, max_mult*np.floor(firlen0*100/delta_cf[index]))

        # Design and tame tales.
        lpf_chan = scipy.signal.firwin(firlen + 1, bp_hf[index]/(fs/2), window=('kaiser', 6.4))
        bpf_len = len(hpf_chan) + len(lpf_chan) - 1
        bpf_chan = np.zeros(bpf_len)

        # Copy hpf_chan in.
        bpf_chan[:len(hpf_chan)] = hpf_chan
        # Convolve two halves of filter to make one.
        bpf_chan = scipy.signal.lfilter(lpf_chan, 1, bpf_chan)
        # No need to ensure 0dB gain at centre otherwise flat recombination does not work.
        # Save filter size and filter for later. 

        # Add bandpass filter and centre frequency to output.
        bpf.append([bpf_chan, bp_cf[index]])

    return bpf

def vocode(sound, low_Hz, high_Hz, nr_channels, 
        vocoder_function='noise', 
        spacing='greenwood', 
        filterbank_type=('butter', 6),
        envelope_method=('butter', 4, 30),
        equalise=True
    ):

    """Vocodes the given sound by filtering it into nr_channels ranging between low_hz and high Hz. 

    The choice of spacing, filterbank type and envelope extraction method depend on the type of 
    experiment that is being implemented. Note that this code is written specifically for the
    psychometric tests incorporated in the toolbox and cannot be easily extended to other experiments.
    This code is not foolproof and comes with absolutely no warranty! 
    """

    def _generate_filler_noise(pulses, f0_contours, time_in_second):
        noise = np.zeros(len(pulses))
        for i in range(len(f0_contours)  - 1):
            if f0_contours[i] == 0 and f0_contours[i + 1] == 0:
                t_start_frames = int(time_in_second[i]*pulses.samplerate_Hz)
                t_end_frames = int(time_in_second[i + 1]*pulses.samplerate_Hz)
                noise[t_start_frames:t_end_frames] = np.random.randn(t_end_frames-t_start_frames)
        noise = Sound(noise, pulses.samplerate_Hz)
        noise.level_dB = pulses.level_dB
        return noise

    def _pad_or_truncate(source, target):
        if len(source) != len(target):
            min_len = min(len(source), len(target))
            new_source = np.zeros(len(target))
            new_source[:min_len] = source[:min_len]
            return new_source
        return source 

    filterbank_type = np.atleast_1d(filterbank_type)

    if spacing == 'greenwood':
        filter_parameters = greenwood_space(low_Hz, high_Hz, nr_channels)
        if filterbank_type[0] == 'fir':
            raise ValueError('FIR filter design does not work with greenwood spacing.')
    elif spacing == 'erb':
        edges = erb_space(nr_channels, high_Hz, low_Hz)
        if filterbank_type[0] != 'fir':
            raise ValueError('ERB spacing can only be combined with FIR filter design (following Hopkins et al. (2010)).')
        filter_parameters = erb_fir_filters(sound, edges, sound.samplerate_Hz)

    vocoded_sound = 0
    for channel_index, filter_params in enumerate(filter_parameters):
        channel = apply_filter(sound, filterbank_type, filter_params)
        channel_level = channel.level_dB
        channel_envelope = apply_envelope_extraction(channel, envelope_method)

        if vocoder_function == 'noise':
            carrier = np.random.randn(len(sound))
            modulated_carrier = carrier*channel_envelope
        elif vocoder_function == 'pass':
            modulated_carrier = channel        
        elif vocoder_function == 'dudley':
            if channel_index == 0:
                carrier_base, f0_contours, time_in_second = generate_f0_pulses(sound, interpolate=False)
            carrier = carrier_base + _generate_filler_noise(carrier_base, f0_contours, time_in_second)
            carrier = _pad_or_truncate(carrier, sound)
            modulated_carrier = carrier*channel_envelope
        elif vocoder_function == 'periodic':
            if channel_index == 0:
                carrier, _, _ = generate_f0_pulses(sound, interpolate=True)
            carrier = _pad_or_truncate(carrier, sound)
            modulated_carrier = carrier*channel_envelope
        else:
            function, parameters = vocoder_function
            modulated_carrier = function(sound, channel_index, filter_params, channel, channel_envelope, *parameters)

        # Reapply the channel filter to remove any out of band modulations.
        vocoded_channel = apply_filter(modulated_carrier, filterbank_type, filter_params)
        if equalise:
            vocoded_channel.level_dB = channel_level
        vocoded_sound += vocoded_channel

    vocoded_sound = Sound(vocoded_sound, sound.samplerate_Hz)

    if equalise:
        vocoded_sound.level_dB = sound.level_dB
    return vocoded_sound