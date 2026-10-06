"""Frame-to-frame stabilizers: discrete hold, roll hysteresis, EMA."""


class Hold:
    """A new value must be seen n_frames consecutive frames to become current."""

    def __init__(self, n_frames, initial=None):
        self.n = n_frames
        self._value = initial
        self._cand = initial
        self._count = 0

    @property
    def value(self):
        return self._value

    def update(self, v):
        if v == self._value:
            self._cand, self._count = v, 0
            return self._value
        if self.n <= 1:
            self._value = v
            return v
        if v == self._cand:
            self._count += 1
        else:
            self._cand, self._count = v, 1
        if self._count >= self.n:
            self._value, self._count = v, 0
        return self._value


class RollZone:
    def __init__(self, minor_deg, deadband_deg):
        self.enter = minor_deg + deadband_deg
        self.leave = minor_deg - deadband_deg
        self._zone = "natural"

    @property
    def zone(self):
        return self._zone

    def update(self, roll_deg):
        r = abs(roll_deg)
        if self._zone == "natural" and r > self.enter:
            self._zone = "flipped"
        elif self._zone == "flipped" and r < self.leave:
            self._zone = "natural"
        return self._zone


class EMA:
    def __init__(self, alpha, initial=0.0):
        self.alpha = alpha
        self._value = float(initial)

    @property
    def value(self):
        return self._value

    def update(self, x):
        self._value += self.alpha * (x - self._value)
        return self._value

    def set(self, x):
        """Jump straight to `x` (a slider moved it; no smoothing)."""
        self._value = float(x)
