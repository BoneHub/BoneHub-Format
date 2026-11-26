# {{TITLE}}: webpage

The webpage of {{TITLE}}: a static site for GitHub Pages that shows an overview of the database (counts, what the
scans hold, its datasets, sex, age, modality and the structures segmented) and a table of every scan, with filters and
a direct download link to each of its files on Hugging Face. Database: {{HF_URL}}

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

After the database changes and its index is rebuilt:

```
python update_data.py
git add data.js
git commit -m "Update data"
git push
```

`update_data.py` reads the index in `../index/` (or the folder you pass it) and needs only pyarrow. It reads
`structures.json` next to it, the part of the body each label belongs to; that file comes from the label map of the
BoneHub-Format package, and `build_webpage` writes it again when the label map changes.

## Change it

`index.html`, `styles.css` and `app.js` come from the template of the BoneHub-Format package, and
`bonehub_format.webpage.build_webpage` fills in the database's name and links. Running it again replaces them, so to
keep edits of your own, make them here and afterwards run only `update_data.py`.
