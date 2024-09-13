def design_steps():
    steps = """
    (1) Consider source support: D1 < L by a factor of 2,3 or more. i.e., if
        your aperture is 10mm, you want a viewing window of 30mm

    (2) Determine sampling regime
        (a) dx > lambda * z / L: TF propagation will work with minor aliasing
        (b) dx > lambda * z / L: Critical sampling, TF approach
        (c) dx > lambda * z / L: IR propagation will work with minor aliasing

    (3) Consider source bandwidth criterion if you are using Fresnel
        propagator. Call bandwidth_criterion() also in this module to
        see a reference table.

    (4) Reconsider source sampling depending on bandwidth criteria and resulting
        artifacts.
    """
    print(steps)


def bandwidth_crieterion():
    criterion = """
+-----------+---------------------------+------------------------+-----------------------+--------------+
| Regime    | Source bandwidth criterion| Approach               | Comments              | Chirp phase  |
|           |                           |                        |                       | sampling     |
|           |                           |                        |                       |(Fresnel only)|
+-----------+---------------------------+------------------------+-----------------------+--------------+
| dx > λz/L | B1 <= 1 / (2dx)           | IR: Periodic copies    | Short z or small λ    | TF: Over     |
|           |                           | TF: Preferred          | Observ plane limited  | IR: Under    |
+-----------+---------------------------+------------------------+-----------------------+--------------+
| dx = λz/L | B1 <= 1 / (2dx) or        | TF and IR identical    | Full use of array     | TF: Critical |
|           | B1 <= L / (2 * lambda *z) |                        | space                 | IR: Critical |
+-----------+---------------------------+------------------------+-----------------------+--------------+
| dx < λz/L | B1 <= L / (2 * lambda *z) | TF: if BW criterion    | Long z or large λ     | TF: Under    |
|           |                           | essesntially met       | space                 | IR: Over     |
|           |                           | IR: Better if bandwidth|                       |              |
|           |                           | criterion violated     | Source bandwidth limit|              |
+-----------+---------------------------+------------------------+-----------------------+--------------+
"""
    print(criterion)


def kernel_differences():
    differences = """
    1. Rayleigh-Sommerfeld Kernel:
       - Exact solution for scalar wave propagation.
       - Works for both near-field and far-field propagation.
       - More computationally intensive, but high accuracy.
       - Use when precision is needed for short or moderate distances.

    2. Fresnel Approximation Kernel:
       - Simplified version of RS, assumes paraxial approximation.
       - Suitable for near to intermediate field propagation.
       - Less computationally expensive, but maintains reasonable accuracy.
       - Use for moderate distances or in optical systems like lenses.

    3. Fraunhofer Approximation Kernel:
       - Far-field approximation, assumes large propagation distance.
       - Direct Fourier transform of the aperture function.
       - Very computationally efficient, but applies only to far-field.
       - Use when dealing with long distances or diffraction patterns.
    """
    print(differences)


def fresnel_number(r: float, z: float, lam: float) -> float:
    """
    Calculates the fresnel number of an aperture

    Parameters:
    r         : radius or halfwidth of aperture
    z         : distance from screen
    lam       : wavelength of light
    """

    return r**2 / (z * lam)


if __name__ == "__main__":
    design_steps()
    bandwidth_crieterion()
