1. Create a branch ``git checkout -b RELEASE_X.Y.Z``
2. Edit ``doc/whats_new/x.y.rst`` to put the date of the release
3. Create a new changelog ``doc/whats_new/x.y+1.rst``, and link it in ``doc/whats_new.rst``
4. Change the ``__version__`` parameter
5. Push and merge the PR
6. create a tag ``git tag X.Y.Z`` and push it ``git push --tags``
7. Go back to dev mode by modifying the what's new with ``Version X.Y+1 -- in dev``, the ``__version__`` and commiting.