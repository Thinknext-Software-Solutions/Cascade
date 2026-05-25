# Studio static frontend

This directory is where the pre-built Next.js frontend (from the
[cascade-studio](https://github.com/Thinknext-Software-Solutions/Cascade-Studio)
repo) gets bundled before each cascade-agent release.

## What's here today

`index.html` is a brand-aligned placeholder that explains Studio's state
and links to the API explorer. It's served whenever the FastAPI app
mounts the static frontend (i.e., when `cascade ui` runs and the real
Next.js build hasn't been copied in yet).

## Updating to the real Next.js build

```bash
# In the cascade-studio repo
cd path/to/cascade-studio
npm install
npm run build       # produces ./out (static export)

# Copy the output here, overwriting the placeholder
rm -rf path/to/cascade-agent/src/cascade/studio/static/*
cp -r out/* path/to/cascade-agent/src/cascade/studio/static/

# Commit the updated bundle in cascade-agent
cd path/to/cascade-agent
git add src/cascade/studio/static/
git commit -m "Bundle Studio UI <version>"
```

The Next.js project must be configured with `output: "export"` in
`next.config.ts` for this to work.

## Why bundle this way?

This keeps `cascade-agent` self-contained as a single pip install. Users
don't need Node.js to run `cascade ui`; the Python wheel ships the
already-built frontend.
