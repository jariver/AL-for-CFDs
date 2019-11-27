class Tuple:
    r"""Tuple Class

    Attributes:
        cid (int): The only id can identify the tuple.
        value_dict (dict): Attribute-value pairs of the tuple, e.g.{'CC': '01', 'AC': '908', 'ZIT': '07974', ...}.
        feature_vec (list): A list of whether the tuple satisfies every predicate. E.g. [0, 1, 0, ...].
        label (int): Whether the tuple violates the CFDs user wants to express. 1 or 0.
        confidence ():
    """
    def __init__(self):
        self.cid = -1
        self.value_dict = dict()
        self.feature_vec = list()
        self.label = None
        self.confidence = 0
