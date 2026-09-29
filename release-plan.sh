#!/usr/bin/env bash
# What a release would publish, and whether the versions agree (RELEASING.md):
#
#   ./release-plan.sh
#
# Lists every artifact that opts into publication with its version, whether Maven Central holds
# it, and whether its code changed since the tag it was released under. Fails when code changed
# under a version Central holds, and when the version an artifact declares is not the one its
# project manages it at or jinfer-bom names.

set -euo pipefail

cd "$(dirname "$0")"
CENTRAL=https://repo1.maven.org/maven2/com/qxotic
BOM=jinfer/jinfer-bom

# Element $1 of POM $2, as the POM states it about itself: outside <parent>, before any content.
declared() {
    sed -e '/<parent>/,/<\/parent>/d' "$2" | sed -n "
        /<\(properties\|dependencies\|dependencyManagement\|build\|modules\|profiles\)>/q
        s|^  <$1>\(.*\)</$1>.*|\1|p" | head -1
}

# The version POM $1 is built with: its own, else its parent's.
version_of() {
    local pom=$1 version
    until version=$(declared version "$pom") && [ -n "$version" ]; do
        pom=$(dirname "$(dirname "$pom")")/pom.xml
    done
    echo "$version"
}

# The version the POMs after $1 manage artifact $1 at, with a property looked up in them.
managed_at() {
    local artifact=$1 version
    shift
    version=$(sed -n "/<dependencyManagement>/,/<\/dependencyManagement>/ {
        /<artifactId>$artifact<\/artifactId>/ { n; s|.*<version>\(.*\)</version>.*|\1|p; }
    }" "$@" | head -1)
    case "$version" in
        '${'*) version=${version:2:-1}; sed -n "s|.*<$version>\(.*\)</$version>.*|\1|p" "$@" | head -1 ;;
        *) echo "$version" ;;
    esac
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

    # Released under its own tag, its project's, or the one of the release of everything.
    code='not released yet'
    for tag in "$artifact-v$version" "$project-v$version" "v$version"; do
        git rev-parse -q --verify "refs/tags/$tag" > /dev/null || continue
        # What ships: src/main, and for jam-native the C sources and their build.
        if git diff --quiet "$tag" -- "$dir/src" "$dir/include" "$dir/cmake" "$dir/CMakeLists.txt" \
            ":(exclude)$dir/src/test"; then
            code="unchanged since $tag"
        else
            code="changed since $tag"
            [ $central = no ] || problem "$artifact changed and $version is on Maven Central: it needs a new version"
        fi
        break
    done
    printf '%-38s %-8s %-11s %s\n' "$artifact" "$version" "$central" "$code"

    for where in "$project/pom.xml pom.xml" "$BOM/pom.xml"; do
        # shellcheck disable=SC2086
        other=$(managed_at "$artifact" $where)
        [ -z "$other" ] || [ "$other" = "$version" ] || problem "$artifact is $version, ${where%% *} says $other"
    done
done < <(git ls-files '*pom.xml' | xargs grep -l '<maven.deploy.skip>false</maven.deploy.skip>' | sort)

echo
echo "to publish, the catalog last:"
[ ${#pending[@]} -gt 0 ] || echo "  nothing: Maven Central holds every version"
for dir in "${pending[@]}"; do [ "$dir" = $BOM ] || echo "  make release-deploy PROJECT=$dir"; done
for dir in "${pending[@]}"; do [ "$dir" != $BOM ] || echo "  make release-deploy PROJECT=$dir"; done

[ $problems -eq 0 ] || { echo "release-plan: $problems problem(s)" >&2; exit 1; }
