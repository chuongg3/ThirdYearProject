'''Class encapsulating compilation flags.
Handles defaults, conversion from/to string, equality comparison'''

_DEFAULTS = {
    'TECHNIQUE': 'baseline',    # type of function merging: 'baseline', 'hyfm', or 'f3m'
    'ALIGNMENT': 'nw',          # type of alignment for hyfm and f3m: 'nw' (Needleman-Wunsch) or 'pa' (Pairwise Alignment)
    'MH_ACROSS': 'true',        # Calculate MinHashes with shingles crossing basic block boundaries
    'F3M_ROWS': '2',            # Numbers of LSH Rows used by f3m
    'F3M_BANDS': '100',         # Numbers of LSH Bands used by fem
    'F3M_RANKDIST': '1.0',      # Maximum fingerprint distance allowed to be selected
    'F3M_ADAPT_BANDS': 'false', # Adaptive number of LSH bands
    'F3M_ADAPT_THR': 'false',   # Adaptive fingerprint distance threshold
    'BUCKET_SIZE_CAP': '100',   # Maximum size of LSH Bucket
    'MATCHER_REPORT': 'false',  # Produce a report on the occupancy of LSH Buckets
    'REPORT': 'false',          # Produce a report on the fingerprint distances and alignment ratios of all valid function pairs
    'PredictAlignment': 'false'
}

# _TECHNIQUES = {
#     'baseline': {'TECHNIQUE': 'baseline'},
#     'hyfm': {'TECHNIQUE': 'hyfm', 'ALIGNMENT': 'pa'},
#     'f3m': {'TECHNIQUE': 'f3m', 'ALIGNMENT': 'pa'},
#     'f3m-predict': {'TECHNIQUE': 'f3m', 'ALIGNMENT': 'pa', 'PredictAlignment':'true'},
#     'f3m-adapt': {'TECHNIQUE': 'f3m', 'ALIGNMENT': 'pa', 'F3M_ADAPT_BANDS': 'true', 'F3M_ADAPT_THR': 'true'}
# }

_TECHNIQUES = {
    'baseline': {'TECHNIQUE': 'baseline'},
    'f3m-nw': {'TECHNIQUE': 'f3m', 'ALIGNMENT': 'nw'},
    'f3m-predict-dotprod-nw': {'TECHNIQUE': 'f3m-predict-dotprod', 'ALIGNMENT': 'nw', 'PredictAlignment':'true'},
    'f3m-predict-attention-nw': {'TECHNIQUE': 'f3m-predict-attention', 'ALIGNMENT': 'nw', 'PredictAlignment':'true'},
    'f3m-pa': {'TECHNIQUE': 'f3m', 'ALIGNMENT': 'pa'},
    'f3m-predict-dotprod-pa': {'TECHNIQUE': 'f3m-predict-dotprod', 'ALIGNMENT': 'pa', 'PredictAlignment':'true'},
    'f3m-predict-attention-pa': {'TECHNIQUE': 'f3m-predict-attention', 'ALIGNMENT': 'pa', 'PredictAlignment':'true'}
}

class Flags(object):
    _global = dict()

    def __init__(self, flags, name=None):
        self.flags = dict(flags)
        self._name = name

    def __eq__(self, other):
        # Check whether the defined properties of self are the same as other's
        for prop, val in self.flags.items():
            if val != other.flags.get(prop, _DEFAULTS[prop]):
                return False

        # Check whether the defined properties of other as the same as self
        for prop, val in other.flags.items():
            if val != self.flags.get(prop, _DEFAULTS[prop]):
                return False

        return True

    def __repr__(self):
        return ' '.join([f'{prop}={self.flags[prop]}' for prop, val in _DEFAULTS.items() if self.flags.get(prop, val) != val or prop == 'TECHNIQUE'])

    def __str__(self):
        return self.__repr__()

    def __hash__(self):
        return hash(self.__repr__())

    def name(self):
        '''Methods returning the technique name if the flags
        match one of the predefined techniques'''
        if self._name:
            return self._name
        for candidate in _TECHNIQUES:
            if self == Flags.from_name(candidate):
                return candidate
        return None

    def mkfile_fmt(self):
        new_flags = dict()
        for prop, val in self._global.items():
            if isinstance(val, bool):
                new_flags[prop.upper()] = 'true' if val else 'false'
            else:
                new_flags[prop.upper()] = val

        for prop, val in self.flags.items():
            if isinstance(val, bool):
                new_flags[prop.upper()] = 'true' if val else 'false'
            else:
                new_flags[prop.upper()] = val
        return new_flags

    def mkfile_opts(self):
        return [f'{prop}={val}'for prop, val in self.mkfile_fmt().items()]


    @classmethod
    def from_str(cls, txt):
        '''Class method translating a string generated
        by __repr__ back into a Flags object'''
        flags = [field for field in txt.split() if field.find('=') >= 0]
        flags = dict([field.split('=') for field in flags])
        return cls(flags)

    @classmethod
    def from_name(cls, name):
        '''Class method translating a technique's name to a Flags object'''
        assert name in _TECHNIQUES
        return cls(_TECHNIQUES[name], name=name)

    @classmethod
    def get_standard(cls):
        '''Class method return Flags objects for all predefined techniques'''
        return [cls(flags, name=name) for name, flags in _TECHNIQUES.items()]

    @classmethod
    def get_report_flags(cls):
        '''Class method return Flags objects for the report run'''
        return [cls({'TECHNIQUE': 'f3m', 'ALIGNMENT': 'pa', 'REPORT': 'true'}), cls({'TECHNIQUE': 'f3m', 'REPORT': 'true'})]

    @classmethod
    def set_globals(cls, global_flags):
        cls._global = global_flags

    @classmethod
    def globals_repr(cls):
        return ' '.join([f'{prop}={val}' for prop, val in cls._global.items()])

    @classmethod
    def from_scratch(cls):
        return cls._global['from_scratch']

    @classmethod
    def llvm_dir(cls):
        return cls._global['llvm_dir']
