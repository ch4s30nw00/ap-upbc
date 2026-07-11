class Evidence:
    def __init__(self, name: str, ls: float, ln: float):
        """
        :param name: 증거(질의)의 이름
        :param ls: Likelihood Ratio for Sufficiency
        :param ln: Likelihood Ratio for Necessity
        """
        if ls <= 0 or ln <= 0:
            raise ValueError(f"LS/LN must be positive: name={name!r}, LS={ls}, LN={ln}")

        self.name = name
        self.ls = ls
        self.ln = ln

        # U(E) = max(|LS - 1|, |1 - LN|)
        self.utility = max(abs(self.ls - 1), abs(1 - self.ln))

    def __repr__(self):
        return f"Evidence(name='{self.name}', LS={self.ls}, LN={self.ln}, U={self.utility:.2f})"
