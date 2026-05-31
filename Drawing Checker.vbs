' 콘솔 창 없이 Drawing Checker 실행 (익스플로러에서 더블클릭)
Dim sh, fso, here
Set sh = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")
here = fso.GetParentFolderName(WScript.ScriptFullName)
sh.CurrentDirectory = here
sh.Run "pythonw """ & here & "\main.py""", 0, False
