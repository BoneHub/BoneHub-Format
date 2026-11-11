# {{TITLE}}: webpage

The webpage of {{TITLE}}: a static site for GitHub Pages that shows an overview of the dataset and searchable tables
of its scans and source datasets. Dataset: {{HF_URL}}

## Publish it

1. Create a GitHub repository for the page, e.g. `<organisation>/<name>`.
2. Push the contents of this folder to it:

   ```
   cd github-webpage
   git init -b main
   git add .
   git commit -m "Dataset webpage"
   git remote add origin https://github.com/<organisation>/<name>.git
   git push -u origin main
   ```

3. In the repository's **Settings → Pages**, set **Source** to *Deploy from a branch*, with branch `main` and folder
   `/ (root)`. The page appears at `https://<organisation>.github.io/<name>/`.

## Update it

After the dataset changes and its index is rebuilt:

```
python update_data.py
git add data.js
git commit -m "Update data"
git push
```

`update_data.py` reads the index in `../index/` (or the folder you pass it) and needs only pyarrow.

## Change it

`index.html`, `styles.css`, `app.js` and `overview.js` come from the template of the BoneHub-Format package, and
`bonehub_format.webpage.build_webpage` fills in the dataset's name and links. Running it again replaces them, so to
keep edits of your own, make them here and afterwards run only `update_data.py`.
