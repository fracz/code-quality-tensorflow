// original filename: 00003048.txt
// after
public class Class00000461Better {
private RefactoringActionHandler getActionHandler(final DataContext dataContext) {
    final MoveAction.MoveProvider moveProvider = MoveAction.MoveProvider.DATA_KEY.getData(dataContext);
    if (moveProvider != null) {
        return moveProvider.getHandler(dataContext);
    } else {
        return RefactoringActionHandlerFactory.getInstance().createMoveHandler();
    }
}
}