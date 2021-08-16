#!/bin/sh

#latesttag=$(git describe --tags `git rev-list --tags --max-count=1`)
#echo checking out ${latesttag}
#git checkout ${latesttag}
git checkout main
#git submodule update --remote
pushd docs/source
make html
#ghp-import -c docs.pymc.io -n -p _build/html/
ghp-import -o -n -p _build/html/
popd
