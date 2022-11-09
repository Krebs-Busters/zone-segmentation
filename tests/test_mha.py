import medpy.io

def test_mha():
    adc = medpy.io.load('./data/test/10000/10000_1000000_adc.mha')
    assert adc[0].shape == (116, 114, 31)