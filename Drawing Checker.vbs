' 콘솔 창 없이 Drawing Checker 실행 (익스플로러에서 더블클릭)
' 파일 연결로 넘어온 파일 경로(있으면 함께 열기)를 인자로 전달
Dim sh, fso, here, fileArg
Set sh = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")
here = fso.GetParentFolderName(WScript.ScriptFullName)
sh.CurrentDirectory = here
fileArg = ""
If WScript.Arguments.Count > 0 Then
    fileArg = " """ & WScript.Arguments(0) & """"
End If
sh.Run "pythonw """ & here & "\main.py""" & fileArg, 0, False
