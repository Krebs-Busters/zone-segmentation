import medpy.io

def test_mha():
    adc = medpy.io.load('./data/test/10000/10000_1000000_adc.mha')
    assert adc[0].shape == (116, 114, 31)
    cor = medpy.io.load('./data/test/10000/10000_1000000_cor.mha')
    assert cor[0].shape == (640, 640, 23)
    hbv = medpy.io.load('./data/test/10000/10000_1000000_hbv.mha')
    assert hbv[0].shape == (116, 114, 31)
    sag = medpy.io.load('./data/test/10000/10000_1000000_sag.mha')
    assert sag[0].shape == (640, 640, 29)
    t2w = medpy.io.load('./data/test/10000/10000_1000000_t2w.mha')
    assert t2w[0].shape == (640, 640, 31)
