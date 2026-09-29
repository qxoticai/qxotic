#!/usr/bin/env bash
# What a release would publish, and whether the versions agree:
#
#   ./release-plan.sh
#
# An artifact takes its project's version unless its own POM declares one: a project is released
# as a whole, and between two such releases an artifact whose code changed is released alone, on
# a version of its own. For every artifact that opts into publication this prints its version,
# whether Maven Central holds it, and whether its code changed since the tag it was released
# under. It fails when
#   - the code of an artifact changed and its version is already on Central: it needs a new one;
#   - an artifact's version, its <artifactId>.version property and its jinfer-bom entry disagree.
# What it lists as "publish" is what `make release-deploy PROJECT=<dir>` stages, one at a time.

set -euo pipefail

ROOT=$(cd "$(dirname "$0")" && pwd)
cd "$ROOT"
BOM=jinfer/jinfer-bom/pom.xml

# The version a POM declares itself: the first one outside its <parent> block, before any content.
own_version() {
    sed -e '/<parent>/,/<\/parent>/d' "$1" |
        sed -n '/<\(properties\|dependencies\|dependencyManagement\|build\|modules\|profiles\)>/q; s|^  <version>\(.*\)</version>.*|\1|p' | head -1
}

# The version an artifact is built with: its own, else its parent's.
version_of() {
    local pom=$1 v
    v=$(own_version "$pom")
    while [ -z "$v" ]; do
        pom=$(dirname "$(dirname "$pom")")/pom.xml
        [ -f "$pom" ] || { echo "?"; return; }
        v=$(own_version "$pom")
    done
    echo "$v"
}

artifact_of() { sed -e '/<parent>/,/<\/parent>/d' "$1" | sed -n 's|^  <artifactId>\(.*\)</artifactId>.*|\1|p' | head -1; }

# The tag an artifact was released under: its own, its project's, or the first release of all.
tag_of() {
    local artifact=$1 project=$2 version=$3 t
    for t in "$artifact-v$version" "$project-v$version" "v$version"; do
        if git rev-parse -q --verify "refs/tags/$t" > /dev/null; then echo "$t"; return; fi
    done
}

bom_entry() { sed -n "/<artifactId>$1<\/artifactId>/{n;s|.*<version>\(.*\)</version>.*|\1|p;}" "$BOM" | head -1; }
property() { sed -n "s|.*<$1.version>\(.*\)</$1.version>.*|\1|p" "$2/pom.xml" pom.xml 2>/dev/null | head -1; }

errors=0
publish=()
printf '%-40s %-8s %-12s %s\n' artifact version 'on Central' code
while IFS= read -r pom; do
    dir=$(dirname "$pom")
    project=${dir%%/*}
    artifact=$(artifact_of "$pom")
    version=$(version_of "$pom")
    default=$(version_of "$project/pom.xml")

    if curl -sfI "https://repo1.maven.org/maven2/com/qxotic/$artifact/$version/$artifact-$version.pom" > /dev/null; then
        central=yes
    else
        central=no
        publish+=("$dir")
    fi

    tag=$(tag_of "$artifact" "$project" "$version")
    sources=("$dir/src/main")
    [ "$artifact" = jam-native ] && sources+=("$dir/src" "$dir/include" "$dir/cmake" "$dir/CMakeLists.txt")
    if [ -z "$tag" ]; then
        code='not released yet'
    elif git diff --quiet "$tag" -- "${sources[@]}"; then
        code="unchanged since $tag"
    else
        code="CHANGED since $tag"
        [ "$central" = yes ] && { code="$code: needs a new version"; errors=$((errors + 1)); }
    fi
    printf '%-40s %-8s %-12s %s\n' "$artifact" "$version" "$central" "$code"

    named=$(property "$artifact" "$project")
    if [ "$version" != "$default" ] && [ "$artifact" != jinfer-bom ] && [ "$artifact" != jinfer-cli ] && [ -z "$named" ]; then
        echo "    no <$artifact.version> property names $version for the rest of $project" >&2
        errors=$((errors + 1))
    fi
    if [ -n "$named" ] && [ "$named" != "$version" ]; then
        echo "    <$artifact.version> says $named" >&2
        errors=$((errors + 1))
    fi
    listed=$(bom_entry "$artifact")
    if [ -n "$listed" ] && [ "$listed" != "$version" ]; then
        echo "    jinfer-bom says $listed" >&2
        errors=$((errors + 1))
    fi
done < <(git ls-files '*pom.xml' | xargs grep -l '<maven.deploy.skip>false</maven.deploy.skip>' | sort)

echo
if [ ${#publish[@]} -eq 0 ]; then
    echo "nothing to publish: every version is on Maven Central"
else
    echo "to publish, the catalog last:"
    for dir in "${publish[@]}"; do [ "$dir" = jinfer/jinfer-bom ] || echo "  make release-deploy PROJECT=$dir"; done
    for dir in "${publish[@]}"; do [ "$dir" = jinfer/jinfer-bom ] && echo "  make release-deploy PROJECT=$dir"; done
fi
[ "$errors" -eq 0 ] || { echo "release-plan: $errors problem(s)" >&2; exit 1; }
