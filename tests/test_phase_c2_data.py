"""Synthetic fixed-input loader contracts; no model forward or project data."""
import csv
import json

import numpy as np
import pytest

import MMCANetV2.phase_c2_data as c2_data
from MMCANetV2.phase_c2_data import C2FixedDataset, sample_id, sha256_file


@pytest.mark.parametrize('scale', [2, 4])
def test_c2_fixed_loader_binds_source_and_arrays(tmp_path, scale, monkeypatch):
    source = tmp_path / 'source.csv'
    with source.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['role', 'source_group_id', 'patch_id'])
        w.writeheader()
        w.writerow({'role':'train', 'source_group_id':'scene-a', 'patch_id':'7'})
        w.writerow({'role':'confirmation', 'source_group_id':'scene-b', 'patch_id':'8'})
    protocol_path = tmp_path / 'protocol.json'
    protocol = {'protocol_id':'mmcanet_phase_c2_screen_20261029_v2',
                'source_manifest_sha256':sha256_file(source)}
    protocol_path.write_text(json.dumps(protocol))
    monkeypatch.setattr(c2_data, 'C2_PROTOCOL_PATH', protocol_path, raising=False)
    monkeypatch.setattr(c2_data, 'C2_PROTOCOL_SHA256', sha256_file(protocol_path), raising=False)
    monkeypatch.setattr(c2_data, 'C2_PROTOCOL_ID', protocol['protocol_id'], raising=False)
    protocol_hash = sha256_file(protocol_path)
    root = tmp_path / 'fixed'
    ident = sample_id('scene-a', '7')
    row = {'sample_id':ident, 'source_group_id':'scene-a', 'patch_id':'7'}
    for name, shape in [('lr',(1,256//scale,256//scale)), ('target',(1,256,256)), ('optical',(3,256,256))]:
        (root/name).mkdir(parents=True)
        path = root/name/(ident+'.npy')
        np.save(path, np.full(shape, 0.25, dtype=np.float32), allow_pickle=False)
        row[name+'_sha256'] = sha256_file(path)
    files = root/'files.csv'
    with files.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(row)); w.writeheader(); w.writerow(row)
    manifest = {'format':'c2_fixed_inputs_v1', 'status':'complete', 'role':'train', 'scale':scale,
                'confirmation_payload_accessed':False,
                'test_set_accessed':False, 'test_set_used':False, 'sample_count':1,
                'source_manifest_sha256':sha256_file(source), 'files_csv_sha256':sha256_file(files),
                'condition_id':'synthetic', 'degradation':{'synthetic':True},
                'protocol_id':protocol['protocol_id'], 'protocol_sha256':protocol_hash}
    (root/'manifest.json').write_text(json.dumps(manifest))
    dataset = C2FixedDataset(root, source, role='train', scale=scale)
    item = dataset[0]
    assert tuple(item['lr'].shape) == (1,256//scale,256//scale)
    assert tuple(item['target'].shape) == (1,256,256)
    assert tuple(item['optical'].shape) == (3,256,256)
    assert item['source_group_id'] == 'scene-a'

    alternate_source = tmp_path / 'source_reordered.csv'
    with alternate_source.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['role', 'source_group_id', 'patch_id'])
        w.writeheader()
        w.writerow({'role':'confirmation', 'source_group_id':'scene-b', 'patch_id':'8'})
        w.writerow({'role':'train', 'source_group_id':'scene-a', 'patch_id':'7'})
    manifest['source_manifest_sha256'] = sha256_file(alternate_source)
    (root/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='C1 来源清单与冻结 C2 协议不匹配'):
        C2FixedDataset(root, alternate_source, role='train', scale=scale)
    manifest['source_manifest_sha256'] = sha256_file(source)
    (root/'manifest.json').write_text(json.dumps(manifest))

    with pytest.raises(ValueError):
        C2FixedDataset(root, source, role='confirmation', scale=scale)
    np.save(root/'lr'/(ident+'.npy'), np.zeros((1,256//scale,256//scale), dtype=np.float32))
    with pytest.raises(ValueError, match='数组哈希不匹配'):
        dataset[0]
    with source.open('a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['role', 'source_group_id', 'patch_id'])
        w.writerow({'role':'development', 'source_group_id':'scene-a', 'patch_id':'9'})
    manifest['source_manifest_sha256'] = sha256_file(source)
    protocol['source_manifest_sha256'] = sha256_file(source)
    protocol_path.write_text(json.dumps(protocol))
    protocol_hash = sha256_file(protocol_path)
    monkeypatch.setattr(c2_data, 'C2_PROTOCOL_SHA256', protocol_hash, raising=False)
    manifest['protocol_sha256'] = protocol_hash
    (root/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='C1 来源场景组跨角色'):
        C2FixedDataset(root, source, role='train', scale=scale)


def test_c2_protocol_file_must_match_frozen_hash(tmp_path, monkeypatch):
    source = tmp_path / 'source.csv'
    source.write_text('role,source_group_id,patch_id\ntrain,scene-a,7\n')
    protocol_path = tmp_path / 'protocol.json'
    protocol = {'protocol_id':'mmcanet_phase_c2_screen_20261029_v2',
                'source_manifest_sha256':sha256_file(source)}
    protocol_path.write_text(json.dumps(protocol))
    pinned_hash = sha256_file(protocol_path)
    monkeypatch.setattr(c2_data, 'C2_PROTOCOL_PATH', protocol_path, raising=False)
    monkeypatch.setattr(c2_data, 'C2_PROTOCOL_SHA256', pinned_hash, raising=False)
    monkeypatch.setattr(c2_data, 'C2_PROTOCOL_ID', protocol['protocol_id'], raising=False)
    protocol['source_manifest_sha256'] = '0' * 64
    protocol_path.write_text(json.dumps(protocol))
    with pytest.raises(ValueError, match='冻结 C2 协议哈希不匹配'):
        c2_data.verify_frozen_c2_source_manifest(source)
