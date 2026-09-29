#!/usr/bin/env bash
# What a release would publish, and whether the versions agree:
#
#   ./release-plan.sh
#
# A project is released as a whole, and its artifacts take its version. Between two releases of a
# project, an artifact whose code changed is released alone, on a version its own POM declares;
# the next release of the project removes those again (RELEASING.md).
#
# For every artifact that opts into publication this prints its version, whether Maven Central
# holds it, and whether its code changed since the tag it was released under. It fails when
#   - code changed and the version is already on Central: the artifact needs a new one;
#   - the version an artifact declares is not the one its project manages it at, or the one
#     jinfer-bom names.
# It ends with what to stage, one `make release-deploy` each, the catalog last.

set -euo pipefail

cd "$(dirname "$0")"
CENTRAL=https://repo1.maven.org/maven2/com/qxotic
BOM=jinfer/jinfer-bom/pom.xml

# What a POM says about itself, outside its <parent> block and before any content: $1 = the
# element, $2 = the POM.
declared() {
    sed -e '/<parent>/,/<\/parent>/d' "$2" | sed -n "
        /<\(properties\|dependencies\|dependencyManagement\|build\|modules\|profiles\)>/q
        s|^  <$1>\(.*\)</$1>.*|\1|p" | head -1
}

# The version an artifact is built with: its own, else its parent's.
version_of() {
    local pom=$1 version
    version=$(declared version "$pom")
    while [ -z "$version" ]; do
        pom=$(dirname "$(dirname "$pom")")/pom.xml
        [ -f "$pom" ] || { echo "release-plan: no version for $1" >&2; exit 1; }
        version=$(declared version "$pom")
    done
    echo "$version"
}

# The version $1 is named at in the dependencyManagement of the POMs that follow, a property
# looked up in the same POMs; empty when none of them manages it.
managed_at() {
    local artifact=$1 version
    shift
    version=$(sed -n "/<dependencyManagement>/,/<\/dependencyManagement>/ {
        /<artifactId>$artifact<\/artifactId>/ { n; s|.*<version>\(.*\)</version>.*|\1|p; }
    }" "$@" | head -1)
    case "$version" in
        '${'*'}')
            version=${version#??}
            version=${version%?}
            sed -n "s|.*<$version>\(.*\)</$version>.*|\1|p" "$@" | head -1
            ;;
        *) echo "$version" ;;
    esac
}

# The tag an artifact was released under: its own, its project's, or the release of everything.
tag_of() {
    local tag
    for tag in "$1-v$3" "$2-v$3" "v$3"; do
        if git rev-parse -q --verify "refs/tags/$tag" > /dev/null; then
            echo "$tag"
            return
        fi
    done
}

problems=0
problem() {
    echo "    $*" >&2
    problems=$((problems + 1))
}

pending=()
printf '%-38s %-8s %-11s %s\n' artifact version 'on Central' code
while IFS= read -r pom; do
    dir=$(dirname "$pom")
    project=${dir%%/*}
    artifact=$(declared artifactId "$pom")
    version=$(version_of "$pom")

    case $(curl -s -o /dev/null -I -w '%{http_code}' "$CENTRAL/$artifact/$version/$artifact-$version.pom") in
        200) central=yes ;;
        404) central=no; pending+=("$dir") ;;
        *) echo "release-plan: cannot ask Maven Central about $artifact $version" >&2; exit 1 ;;
    esac

    # What ships: the Java sources, and for jam-native the C sources and their build as well.
    sources=("$dir/src/main")
    [ "$artifact" != jam-native ] ||
        sources=("$dir/src" "$dir/include" "$dir/cmake" "$dir/CMakeLists.txt" ":(exclude)$dir/src/test")
    tag=$(tag_of "$artifact" "$project" "$version")
    if [ -z "$tag" ]; then
        code='not released yet'
    elif git diff --quiet "$tag" -- "${sources[@]}"; then
        code="unchanged since $tag"
    else
        code="changed since $tag"
    fi
    printf '%-38s %-8s %-11s %s\n' "$artifact" "$version" "$central" "$code"

    case "$central $code" in
        'yes changed'*) problem "its code changed and $version is on Maven Central: it needs a new version" ;;
    esac
    managed=$(managed_at "$artifact" "$project/pom.xml" pom.xml)
    [ -z "$managed" ] || [ "$managed" = "$version" ] || problem "its project manages it at $managed"
    named=$(managed_at "$artifact" "$BOM")
    [ -z "$named" ] || [ "$named" = "$version" ] || problem "jinfer-bom names it at $named"
done < <(git ls-files '*pom.xml' | xargs grep -l '<maven.deploy.skip>false</maven.deploy.skip>' | sort)

echo
if [ ${#pending[@]} -eq 0 ]; then
    echo "nothing to publish: Maven Central holds every version"
else
    echo "to publish, the catalog last:"
    for dir in "${pending[@]}"; do
        [ "$dir" = "$(dirname $BOM)" ] || echo "  make release-deploy PROJECT=$dir"
    done
    for dir in "${pending[@]}"; do
        [ "$dir" != "$(dirname $BOM)" ] || echo "  make release-deploy PROJECT=$dir"
    done
fi
if [ "$problems" -ne 0 ]; then
    echo "release-plan: $problems problem(s)" >&2
    exit 1
fi
