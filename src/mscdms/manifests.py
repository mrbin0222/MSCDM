"""Decode the fixed MSCDM sample catalog into runtime manifest interfaces.

All text values are looked up verbatim; this module never generates prompts.
The catalog is read-only. Materialized files belong in a local data directory.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Iterator

DEFAULT_CATALOG = Path(__file__).resolve().parents[2] / 'configs' / 'manifests'
PROFILES = ('samples', 'reconstruction', 'evaluator_manifest', 'foreground_samples')


def read_jsonl(path: Path) -> list[dict]:
    with Path(path).open(encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


class ManifestCatalog:
    def __init__(self, root: str | Path = DEFAULT_CATALOG):
        self.root = Path(root)
        self.description = json.loads((self.root / 'catalog.json').read_text(encoding='utf-8'))
        if self.description.get('schema') != 'mscdm.manifest_catalog.v1':
            raise ValueError('Unsupported manifest catalog schema')
        self.rows = read_jsonl(self.root / 'samples.jsonl')
        self.tables = self.description['tables']
        self.validate()

    def validate(self) -> None:
        if len(self.rows) != self.description['counts']['samples']:
            raise ValueError('Sample count differs from catalog')
        if len({r['sample_id'] for r in self.rows}) != len(self.rows):
            raise ValueError('Duplicate sample ID')
        for profile in PROFILES:
            order = self.description['profile_order'][profile]
            if len(order) != self.description['counts'][profile] or len(order) != len(set(order)):
                raise ValueError(f'Invalid {profile} membership/order')
            if any(type(i) is not int or not 0 <= i < len(self.rows) for i in order):
                raise ValueError(f'Invalid {profile} sample index')
        for row in self.rows:
            if row['split'] not in ('train', 'val', 'test'):
                raise ValueError('Unknown data split')
            for field, table in [('source_id', 'sources'), ('parent_id', 'parents'),
                                 ('attributes_id', 'attributes'), ('labels_id', 'labels')]:
                self._lookup(table, row[field])
            for key in ('text_ids', 'evaluation_text_ids'):
                self._texts(row[key])
            if 'foreground' in row:
                self._texts(row['foreground']['text_ids'])
                self._lookup('foreground_controls', row['foreground']['controls_id'])
        expected = {i for i, row in enumerate(self.rows) if 'foreground' in row}
        if set(self.description['profile_order']['foreground_samples']) != expected:
            raise ValueError('Foreground membership differs from catalog')

    def _lookup(self, table: str, index: int):
        if type(index) is not int or not 0 <= index < len(self.tables[table]):
            raise ValueError(f'Invalid {table} index {index!r}')
        return self.tables[table][index]

    def _texts(self, indexes: list[int]) -> dict:
        if len(indexes) != 3:
            raise ValueError('Expected three text fields')
        return dict(zip(self.description['text_fields'], (self._lookup('texts', i) for i in indexes)))

    def _path(self, kind: str, row: dict, source: dict) -> str:
        return self.description['derived_paths'][kind].format(**row, **source)

    def _base(self, row: dict) -> dict:
        source = self._lookup('sources', row['source_id'])
        return {'sample_id': row['sample_id'], 'split': row['split'],
                'source_name': source['source_name'], 'object_type': source['object_type'],
                **self._texts(row['text_ids']), 'instance_id': row['instance_id']}

    def _reconstruction(self, row: dict) -> dict:
        return {'sample_id': row['sample_id'], 'split': row['split'],
                **copy.deepcopy(self._lookup('sources', row['source_id'])),
                **copy.deepcopy(self._lookup('parents', row['parent_id'])),
                'instance_id': row['instance_id'], 'bbox_xywh': list(row['bbox_xywh']),
                'attributes': copy.deepcopy(self._lookup('attributes', row['attributes_id']))}

    def iter_profile(self, profile: str) -> Iterator[dict]:
        if profile not in PROFILES:
            raise ValueError(f'Unknown profile {profile}')
        for index in self.description['profile_order'][profile]:
            row = self.rows[index]
            source = self._lookup('sources', row['source_id'])
            if profile == 'samples':
                yield self._base(row)
            elif profile == 'reconstruction':
                yield self._reconstruction(row)
            elif profile == 'evaluator_manifest':
                texts = self._texts(row['evaluation_text_ids'])
                yield {'sample_id': row['sample_id'], 'split': row['split'],
                       'source_name': source['source_name'], 'object_type': source['object_type'],
                       'image_path': self._path('image', row, source), **texts,
                       **{'canonical_' + k + '_prompt': v for k, v in texts.items()},
                       **copy.deepcopy(self._lookup('labels', row['labels_id']))}
            else:
                foreground = row['foreground']
                yield {'sample_id': row['sample_id'], 'split': row['split'],
                       'source_name': source['source_name'], 'object_type': source['object_type'],
                       'image_path': self._path('image', row, source),
                       'target_mask_path': self._path('target_mask', row, source),
                       **self._texts(foreground['text_ids']),
                       'attributes': copy.deepcopy(self._lookup('attributes', row['attributes_id'])),
                       'control_labels': copy.deepcopy(self._lookup('foreground_controls', foreground['controls_id'])),
                       'morphology_mask_surrogate': {
                           'enabled': foreground['surrogate_enabled'],
                           'source_id': foreground['surrogate_source_id'],
                           'target_mask_path': self._path('surrogate_mask', row, source),
                           'target_morphology': foreground['target_morphology']}}

    def iter_training(self) -> Iterator[dict]:
        for row in self.rows:
            source = self._lookup('sources', row['source_id'])
            yield {**self._reconstruction(row), **self._base(row),
                   'image_path': self._path('image', row, source),
                   'target_mask_path': self._path('target_mask', row, source)}

    def iter_masks(self) -> Iterator[dict]:
        for row in self.rows:
            source = self._lookup('sources', row['source_id'])
            yield {'sample_id': row['sample_id'], 'split': row['split'],
                   'source_name': source['source_name'],
                   'target_mask_path': self._path('target_mask', row, source)}

    def materialize(self, output_dir: str | Path) -> dict:
        output_dir = Path(output_dir).resolve()
        if output_dir == self.root.resolve() or self.root.resolve() in output_dir.parents:
            raise ValueError('Runtime manifests must be generated outside the source catalog')
        outputs = {}
        destinations = {'samples': 'samples.jsonl', 'reconstruction': 'reconstruction.jsonl',
                        'evaluator_manifest': 'control_metrics_v2/evaluator_manifest.jsonl',
                        'foreground_samples': 'foreground_samples.jsonl'}
        for profile, relative in destinations.items():
            outputs[profile] = _write_rows(output_dir / relative, self.iter_profile(profile))
        outputs['training'] = _write_rows(output_dir / 'manifest_hybrid.jsonl', self.iter_training())
        outputs['masks'] = _write_rows(output_dir / 'mask_manifest.jsonl', self.iter_masks())
        _write_bytes(output_dir / 'control_metrics_v2/label_vocab.json', (self.root / 'label_vocab.json').read_bytes())
        for name in ('validation_diagnostics_50.jsonl', 'fixed_geometry_test_180.jsonl', 'auxiliary_test_200.jsonl'):
            _write_bytes(output_dir / 'conditions' / name, (self.root / name).read_bytes())
        return outputs


def _write_bytes(path: Path, content: bytes) -> None:
    if path.exists():
        if path.read_bytes() != content:
            raise FileExistsError(f'Preserve existing file with different content: {path}')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name('.' + path.name + '.tmp')
    try:
        with temporary.open('xb') as handle:
            handle.write(content)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_rows(path: Path, rows: Iterator[dict]) -> dict:
    content = ''.join(json.dumps(r, ensure_ascii=False, separators=(',', ':')) + '\n' for r in rows).encode('utf-8')
    _write_bytes(path, content)
    return {'path': str(path), 'sha256': hashlib.sha256(content).hexdigest(),
            'rows': content.count(b'\n'), 'bytes': len(content)}
