Contributing to CS328 Notes

Some content taken from scikit-learn contributing document.

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
8. Create a feature branch to hold your development changes:
```bash
git checkout -b my_feature
```
and start making changes. Always use a feature branch. It’s good practice to never work on the `main` branch!
9. Develop the feature on your feature branch on your computer, using Git to do the version control. When you’re done editing, add changed files using `git add` and then `git commit`:
```bash
git add modified_files
git commit
```
to record your changes in Git, then push the changes to your GitHub account with:
```bash
git push -u origin my_feature
```
10. Follow below instructions to create a pull request from your fork.
It is often helpful to keep your local feature branch synchronized with the latest changes of the main CS328-Notes repository:
```bash
git fetch upstream
git merge upstream/main
```
Subsequently, you might need to solve the conflicts. You can refer to the [Git documentation related to resolving merge conflict using the command line](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/addressing-merge-conflicts/resolving-a-merge-conflict-using-the-command-line).

## Adding Pages to CS328-Notes book
