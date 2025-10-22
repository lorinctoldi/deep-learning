$Url = "" # get the download URL from Kaggle
$OutFile = "C:\Downloads\train_v2.zip"

Start-BitsTransfer -Source $Url -Destination $OutFile -DisplayName "KaggleDownload" -Description "Downloading Kaggle dataset" -TransferType Download -RetryInterval 60 -RetryTimeout 86400
