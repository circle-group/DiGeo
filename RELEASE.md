1. Create a branch ``git checkout -b RELEASE_X.Y.Z``
2. Edit ``doc/whats_new/x.y.rst`` to put the date of the release
3. Change the ``__version__`` parameter
4. Create a PR ```RELEASE X.Y.Z`` and merge the PR once tests pass
5. create a tag ``git tag vX.Y.Z`` on main and push it ``git push --tags``
6. Create a new changelog ``doc/whats_new/x.y+1.rst``, and link it in ``doc/whats_new.rst``
7. Edit the ``__version__`` to ``X.Y+1dev`` and commit the changes.
