"""Fixed-input preservation and runtime manifest interface checks."""
import hashlib,json,tempfile,unittest,sys,subprocess
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from mscdms.manifests import ManifestCatalog,PROFILES,read_jsonl
ROOT=Path(__file__).resolve().parents[1]

def digest(rows):
    h=hashlib.sha256()
    for row in rows:h.update((json.dumps(row,sort_keys=True,ensure_ascii=False,separators=(',',':'))+'\n').encode())
    return h.hexdigest()

class CatalogContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.catalog=ManifestCatalog(ROOT/'configs/manifests')
    def test_all_fixed_profiles_match_before_compaction(self):
        for profile in PROFILES:
            self.assertEqual(digest(self.catalog.iter_profile(profile)),self.catalog.description['profile_semantic_sha256'][profile])
    def test_materialization_and_original_profile_order(self):
        with tempfile.TemporaryDirectory()as tmp:
            outputs=self.catalog.materialize(tmp)
            for profile in PROFILES:
                self.assertEqual(digest(read_jsonl(Path(outputs[profile]['path']))),self.catalog.description['profile_semantic_sha256'][profile])
            self.assertEqual(outputs['training']['rows'],37500)
            self.assertEqual(outputs['masks']['rows'],37500)
            self.assertTrue((Path(tmp)/'control_metrics_v2/label_vocab.json').is_file())
            for name in ['validation_diagnostics_50.jsonl','fixed_geometry_test_180.jsonl','auxiliary_test_200.jsonl']:
                self.assertEqual((Path(tmp)/'conditions'/name).read_bytes(),(ROOT/'configs/manifests'/name).read_bytes())
    def test_preserve_existing_output_and_catalog(self):
        with self.assertRaises(ValueError):self.catalog.materialize(ROOT/'configs/manifests')
        with tempfile.TemporaryDirectory()as tmp:
            target=Path(tmp)/'samples.jsonl';target.write_text('author-edited content\n')
            with self.assertRaises(FileExistsError):self.catalog.materialize(tmp)
            self.assertEqual(target.read_text(),'author-edited content\n')
    def test_decoded_rows_do_not_share_mutable_attributes(self):
        first=next(self.catalog.iter_profile('reconstruction'));saved=first['attributes'].copy();first['attributes']['morphology']='changed'
        again=next(self.catalog.iter_profile('reconstruction'));self.assertEqual(again['attributes'],saved)
    def test_attach_actual_released_text(self):
        expected=next(self.catalog.iter_profile('samples'))
        with tempfile.TemporaryDirectory()as tmp:
            source=Path(tmp)/'input.jsonl';out=Path(tmp)/'output.jsonl'
            source.write_text(json.dumps({'sample_id':expected['sample_id'],'split':expected['split'],'subject':'wrong','image_path':'example.png'})+'\n')
            result=subprocess.run([sys.executable,str(ROOT/'scripts/data/attach_released_text.py'),'--input',str(source),'--output',str(out),'--catalog',str(ROOT/'configs/manifests')],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            row=read_jsonl(out)[0]
            for k in ['subject','state','background']:self.assertEqual(row[k],expected[k])
            self.assertEqual(row['image_path'],'example.png')
if __name__=='__main__':unittest.main()
