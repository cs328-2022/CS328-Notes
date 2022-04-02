# Contributing to CS328 Notes

A portion of content taken from scikit-learn contributing document.

## Video Resources
- Useful Playlist on Youtube: [Playlist](https://www.youtube.com/playlist?list=PLBKcU7Ik-ir-b1fwjNabO3b8ebs9ez5ga)
- Crash Course in Contributing to Scikit-Learn & Open Source Projects: [Video](https://www.youtube.com/watch?v=5OL8XoMMOfA), [Transcript](https://github.com/data-umbrella/event-transcripts/blob/main/2020/05-andreas-mueller-contributing.md)
- Example of Submitting a Pull Request to scikit-learn: [Video](https://www.youtube.com/watch?v=PU1WyDPGePI), [Transcript](https://github.com/data-umbrella/event-transcripts/blob/main/2020/06-reshama-shaikh-sklearn-pr.md)

## How to Contribute:
1. [Create an account](https://github.com/) on GitHub if you do not already have one.
2. Fork the [project repository](https://github.com/cs328-2022/CS328-Notes): click on the ‘Fork’ button near the top of the page. This creates a copy of the code under your account on the GitHub user account. For more details on how to fork a repository see [this guide](https://docs.github.com/en/get-started/quickstart/fork-a-repo).
3. Clone your fork of the CS328-Notes repo from your GitHub account to your local disk:
    ```bash
    git clone git@github.com:YourLogin/CS328-Notes.git  # add --depth 1 if your connection is slow
    cd CS328-Notes
    ```
4. Install the dependencies for building jupyter-book:
    ```bash
    pip install -r requirements.txt
    ```
5. Add the `upstream` remote. This saves a reference to the main CS328-Notes repository, which you can use to keep your repository synchronized with the latest changes:
    ```bash
    git remote add upstream git@github.com:cs328-2022/CS328-Notes.git
    ```
6. Check that the upstream and origin remote aliases are configured correctly by running git remote -v which should display:
    ```bash
    origin  git@github.com:YourLogin/CS328-Notes.git (fetch)
    origin  git@github.com:YourLogin/CS328-Notes.git (push)
    upstream        git@github.com:cs328-2022/CS328-Notes.git (fetch)
    upstream        git@github.com:cs328-2022/CS328-Notes.git (push)
    ```
7. Synchronize your `main` branch with the `upstream/main` branch, more details on [GitHub Docs](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork):
    ```bash
    git checkout main
    git fetch upstream
    git merge upstream/main
    ```
    In case you face error `Permission denied (publickey)`, you need to [add SSH key to your GitHub account](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account).
8. Create a feature branch to hold your development changes:
    ```bash
    git checkout -b my_feature
    ```
    and start making changes. Always use a feature branch. It’s good practice to never work on the `main` branch!
9. Install pre-commit to run code style checks before each commit:
    ```bash
    pip install pre-commit
    pre-commit install
    ```
10. Develop the feature on your feature branch on your computer, using Git to do the version control. When you’re done editing, add changed files using `git add` and then `git commit`:
    ```bash
    git add modified_files
    git commit
    ```
    to record your changes in Git, then push the changes to your GitHub account with:
    ```bash
    git push -u origin my_feature
    ```
11. Follow below instructions to create a pull request from your fork.
    It is often helpful to keep your local feature branch synchronized with the latest changes of the main CS328-Notes repository:
    ```bash
    git fetch upstream
    git merge upstream/main
    ```
    Subsequently, you might need to solve the conflicts. You can refer to the [Git documentation related to resolving merge conflict using the command line](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/addressing-merge-conflicts/resolving-a-merge-conflict-using-the-command-line).

## Adding Pages to CS328-Notes book
1. Add the Markdown/IPYNB file in the `CS328-Notes/notebooks` directory.
2. In case the Markdown/IPYNB file import images/gifs/data-files, make sure to place the files in a new folder in assets directory `CS328-Notes\assets\<same-name-as-of-IPYNB-MD-file>`.
3. Add the filename to the Table to Content file `CS328-Notes/_toc.yml`.
4. Build the jupyter-book on your local machine using `jupyter-book build CS328-Notes`. The html output will be placed in `CS328-Notes/_build/html` and is for your reference only. The published html pages are generated automatically on github.

## Using MyST markdown features
1. Jupyter Book supports MyST markdown. Refer [Here](https://jupyterbook.org/content/myst.html).
2. Refer [Here](https://jupyterbook.org/content/index.html) for a more broad overview.
3. Please refer [here](https://jupyterbook.org/content/executable/index.html) as well.
4. Use [`Sphinx-proof` syntax](https://sphinx-proof.readthedocs.io/en/latest/syntax.html) for Proofs, algorithms, properties, lemma, etc.

## Passing Checks on PR and preview
1. Make sure all the check pass in the Pull Request.
2. You can preview the temporary build of the pull request on web by clicking on `Details` link of `ci/circleci:build_jupyter_book artifact` check. This is how exactly the E-book would look once the pull request is merged.
3. Resolve the merge conflicts in PR, if any.

## General Instructions
1. Please add captions to figures/images, wherever seems suitable.
2. For each external resource (image, content, etc.), please give due credit to the original source.
3. All mathematical symbols (like x, y, d, C, set A) in the text needs to be inside `$ $`. This makes easy to differentiate these symbols from general text.
4. In case, you are importing CSV or other data files in your code, put these files in corresponding assets folder. Then import the file in program from the assets directory.
5. We can incorporate features from [Sphinx-Proof](https://sphinx-proof.readthedocs.io/en/latest/syntax.html) extension for writing Algorithms, theorems, Lemmas, etc. The extension is enabled in the notebook.


## Pull Request Checklist
Before a PR can be merged, it needs to be approved by a TA. Please prefix the title of your pull request with [MRG] if the PR is complete and should be subjected to a detailed review. An incomplete contribution – where you expect to do more work before receiving a full review – should be prefixed [WIP] (to indicate a work in progress) and changed to [MRG] when it matures, before the assigned deadline with all checks passing and no merge conflicts.

1. Give your pull request a helpful title

   `[WIP or MRG] CS328 Submission : Group <Group No.>`
2. Please make sure the name of md/ipynb files is in format `yyyy_mm_dd_topic_name`, where date is of the day when the corresponding lecture was conducted.
3. In each markdown/notebook file, please mention the names of all the group members in footer tag. At the end of each page:
    ```html
    <footer>
        Author(s): Member1 Name, Member2 Name, Member3 Name
    </footer>
    ```
4. Follow the PR template. The template will automatically open when creating a PR.
5. Make sure all the checks are passing with no merge conflicts.
