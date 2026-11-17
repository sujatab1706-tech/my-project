from PIL import Image

from rules import is_white_background, white_border_fraction


def test_pure_white_is_white():
    assert white_border_fraction(Image.new("RGB", (50, 50), "white")) == 1.0


def test_colour_is_not_white():
    assert not is_white_background(Image.new("RGB", (50, 50), (200, 30, 30)))


def test_tolerance():
    img = Image.new("RGB", (50, 50), (245, 245, 245))
    assert is_white_background(img)                         # default tol=12 accepts 245
    assert white_border_fraction(img, tol=6) == 0.0         # strict tol=6 rejects it
