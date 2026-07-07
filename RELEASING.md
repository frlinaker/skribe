# Releasing skribe

How to cut a new release to GitHub and PyPI. Versioning follows
[SemVer](https://semver.org/); the single source of truth is
`skribe/version.py`.

## 1. Prepare the release commit

1. Bump `__version__` in `skribe/version.py`.
2. Add a dated section to `CHANGELOG.md` (Added / Changed / Fixed).
3. Update "Current stable release" in `ROADMAP.md`.
4. Commit. The pre-commit gate runs `black` and the full live-LLM test suite
   (using `gpt-5.4-mini` via `tests/conftest.py`), so `OPENAI_API_KEY` must be
   set in the environment.

```bash
git add skribe/version.py CHANGELOG.md ROADMAP.md
git commit -m "Release vX.Y.Z: <summary>"
```

## 2. Tag and push to GitHub

```bash
git tag -a vX.Y.Z -m "vX.Y.Z — <summary>"
git push origin main
git push origin vX.Y.Z
```

## 3. Build the distributions

Build into a version-specific directory so only that release's artifacts are
present at upload time (avoids accidentally re-uploading older builds):

```bash
.venv/bin/python -m build --outdir release-vX.Y.Z
```

`MANIFEST.in` keeps local-only / secret-bearing files (`.env`, `.envrc`,
`.cursorrules`, `.claude/`) out of the source distribution, and
`setup.py` (`find_packages(exclude=[...])`) ensures only the `skribe`
package is shipped — not `tests/` or `examples/`.

## 4. Validate before uploading

```bash
.venv/bin/twine check release-vX.Y.Z/*
```

Optional sanity check that the wheel ships only `skribe` and no secrets:

```bash
.venv/bin/python - <<'PY'
import zipfile, glob
z = zipfile.ZipFile(sorted(glob.glob('release-vX.Y.Z/*.whl'))[-1])
tl = next(n for n in z.namelist() if n.endswith('top_level.txt'))
print("top_level:", z.read(tl).decode().split())
print("ships tests/:", any(n.startswith('tests/') for n in z.namelist()))
PY
```

## 5. Upload to PyPI

Credentials live in `~/.pypirc` under the default `[pypi]` section
(`username = __token__`, `password = pypi-...` project-scoped token), so no
flags are needed:

```bash
.venv/bin/twine upload release-vX.Y.Z/*
```

The token is project-scoped to `skribe`, so this default only authorizes
uploading this package. **Never commit the token or `~/.pypirc`.** If a token is
ever exposed (e.g. pasted on a command line), revoke it at
<https://pypi.org/manage/account/token/> and issue a fresh one.

## 6. Verify it is live

```bash
.venv/bin/python - <<'PY'
import urllib.request, json
d = json.load(urllib.request.urlopen("https://pypi.org/pypi/skribe/json", timeout=20))
print("latest on PyPI:", d["info"]["version"])
PY
```

Re-running `twine upload` on already-published, byte-identical files is a
harmless no-op — PyPI artifacts are immutable, so a release can never be
silently overwritten.
