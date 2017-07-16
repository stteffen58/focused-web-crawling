from collections import defaultdict
import sys


class Rule:
    def __init__(self):
        self.c = defaultdict(str)
        self.a = defaultdict(str)
        self.delta_keys = set()
        self.delta_key_values = set()
        self.support = 0

    def __repr__(self):
        return str((self.c,self.a))

    def __str__(self):
        return str((self.c,self.a))

    def __gt__(self, other):
        if self.support >= other.support:
            return True
        return False

    def __lt__(self, other):
        if self.support < other.support:
            return True
        return False

    def __eq__(self, otherRule):
        """Override the default Equals behavior"""
        c_ = otherRule.c
        keyset_c = set.intersection(set(self.c.keys()),set(c_.keys()))
        if len(keyset_c) != len(self.c.keys()):
            return False

        a_ = otherRule.a
        keyset_a = set.intersection(set(self.a.keys()), set(a_.keys()))
        if len(keyset_a) != len(self.a.keys()):
            return False

        for k in keyset_c:
            if self.c.get(k,'') != c_.get(k,''):
                return False

        for k in keyset_a:
            if self.a.get(k,'') != a_.get(k,''):
                return False

        return True