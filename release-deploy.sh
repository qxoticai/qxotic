#!/usr/bin/env bash
# Stage ONE project's artifacts for Maven Central:
#
#   ./release-deploy.sh jinfer          # check, then deploy
#   ./release-deploy.sh jinfer --check  # check only
#
# Each project releases on its own version, so a release is one project's artifacts and nothing
# else. Deploying from the repository root would restage every project that opts into publication,
# including the ones whose version is already on Central, and Central rejects a coordinate it
# already holds, taking the whole bundle with it. This builds only the named project's reactor, and
# refuses before uploading anything if a coordinate in it is already published.
#
# The other projects must be installed first (`mvn -Prelease install -DskipTests`), since this
# reactor resolves them from the local repository. Publication stays manual in Central
# (autoPublish=false). Honors MAVEN / MAVEN_FLAGS like the Makefiles.

set -euo pipefail

ROOT=$(cd "$(dirname "$0")" && pwd)
PROJECT=${1:-}
[ -n "$PROJECT" ] && [ -f "$ROOT/$PROJECT/pom.xml" ] || {
    echo "usage: $(basename "$0") <project> [--check]   (a directory with a pom.xml: jinfer, jota, gguf, ...)" >&2
    exit 2
}
CHECK_ONLY=${2:-}
POM=$ROOT/$PROJECT/pom.xml
# unquoted MAVEN_FLAGS: word-splitting intended, same convention as the Makefiles
MVN="${MAVEN:-mvn} ${MAVEN_FLAGS:-}"
EVALUATE=org.apache.maven.plugins:maven-help-plugin:3.5.2:evaluate

# shellcheck disable=SC2086
VERSION=$($MVN -q -B -f "$POM" $EVALUATE -Dexpression=project.version -DforceStdout 2>/dev/null | tail -1)
case "$VERSION" in
    ''|*' '*) echo "deploy: could not determine $PROJECT's version (got '$VERSION')" >&2; exit 1 ;;
esac

# Publication is opt-in: a module stages when its own POM turns maven.deploy.skip off, the parent
# defaulting it on. That property is what the release profile hands to central-publishing.
STAGED=$(while IFS= read -r pom; do
    # the module's own artifactId: the first one after the parent block
    sed -n '/<\/parent>/,$ s|.*<artifactId>\(.*\)</artifactId>.*|\1|p' "$pom" | head -1
done < <(find "$ROOT/$PROJECT" -name pom.xml -not -path '*/target/*' -print0 |
    xargs -0 grep -l '<maven.deploy.skip>false</maven.deploy.skip>') | sort)
[ -n "$STAGED" ] || { echo "deploy: no module of $PROJECT opts into publication" >&2; exit 1; }

echo "==> $PROJECT $VERSION stages:" $STAGED

PUBLISHED=""
for artifact in $STAGED; do
    url=https://repo1.maven.org/maven2/com/qxotic/$artifact/$VERSION/$artifact-$VERSION.pom
    if curl -sfI "$url" > /dev/null; then PUBLISHED="$PUBLISHED $artifact"; fi
done
if [ -n "$PUBLISHED" ]; then
    echo "deploy: already on Maven Central at $VERSION:$PUBLISHED" >&2
    echo "deploy: bump $PROJECT (its root POM and $PROJECT.version in the root POM) or release another project" >&2
    exit 1
fi
echo "==> none of them is on Maven Central yet"
[ "$CHECK_ONLY" = "--check" ] && exit 0

# shellcheck disable=SC2086
$MVN -B -f "$POM" -Prelease deploy
echo "==> staged; approve the deployment in Central to publish it"
