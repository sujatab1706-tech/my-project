import io

from PIL import Image, ImageDraw


def png_bytes(background, size=(300, 300)):
    img = Image.new("RGB", size, background)
    ImageDraw.Draw(img).rectangle([100, 100, 200, 200], fill=(20, 20, 20))  # the "product"
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_white_background_is_not_nwb(client):
    r = client.post("/predict", files={"file": ("white.png", png_bytes((255, 255, 255)), "image/png")})
    assert r.status_code == 200
    body = r.json()
    assert body["nwb"]["rule_flag"] is False
    assert body["nwb"]["flag"] is False


def test_off_white_background_is_nwb(client):
    # RGB 240 looks white to people but breaks Amazon's pure-white rule
    r = client.post("/predict", files={"file": ("offwhite.png", png_bytes((240, 240, 240)), "image/png")})
    assert r.json()["nwb"]["flag"] is True


def test_response_shape(client):
    body = client.post("/predict", files={"file": ("x.png", png_bytes((255, 255, 255)), "image/png")}).json()
    for key in ["tlgw", "nwb", "compliant", "image", "latency_ms"]:
        assert key in body
    assert 0 <= body["tlgw"]["probability"] <= 1
    assert body["tlgw"]["flag"] is True  # fake model says 0.5, threshold 0.4
    assert body["compliant"] is False


def test_not_an_image(client):
    r = client.post("/predict", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert r.status_code == 400


def test_predictions_are_logged(client):
    client.post("/predict", files={"file": ("x.png", png_bytes((255, 255, 255)), "image/png")})
    stats = client.get("/stats").json()
    assert stats["predictions"] >= 1
    assert 0 <= stats["tlgw_flag_rate"] <= 1
