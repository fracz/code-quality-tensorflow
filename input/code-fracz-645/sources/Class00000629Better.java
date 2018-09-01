// original filename: 00027825.txt
// before
public class Class00000629Better {
/**
 * Gets list of workspaces which has given namespace
 *
 * <p>Returned workspaces have either {@link WorkspaceStatus#STOPPED} status
 * or status defined by their runtime instances(if those exist).
 *
 * @param namespace
 * the namespace to find workspaces
 * @return the list of workspaces or empty list if no matches
 * @throws NullPointerException
 * when {@code namespace} is null
 * @throws ServerException
 * when any server error occurs while getting workspaces with {@link WorkspaceDao#getByNamespace(String)}
 */
public List<WorkspaceImpl> getByNamespace(String namespace) throws ServerException {
    requireNonNull(namespace, "Required non-null namespace");
    final List<WorkspaceImpl> workspaces = workspaceDao.getByNamespace(namespace);
    workspaces.forEach(this::normalizeState);
    return workspaces;
}
}