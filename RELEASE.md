1. Create a branch ``git checkout -b RELEASE_X.Y.Z``
2. Edit ``doc/whats_new.rst`` to put the date of the release
3. Change the ``__version__`` parameter
4. Create a PR ```RELEASE X.Y.Z`` and merge the PR once tests pass
5. create a tag ``git tag X.Y.Z`` and push it ``git push --tags``
6. Go back to dev mode by modifying the what's new with ``Version X.Y+1 -- in dev``,
   the ``__version__`` and commiting.