[im,map]=imread('mio1.jpg');
im=im2double(im);
[row,col]=size(im);

%  number of clusters
nc=4;

%  initial random cluster centroids
cs=rand(nc,1);
pcs=cs;

%  number of iteration 
T=50;
t=0;
D=zeros(row,col,nc);
tsmld=[];
eps=1.e-5;
cmx=1;

while (t<T && cmx>eps)
    %Distance between centroids and image's pixel
    for c=1:nc
        D(:,:,c)=(im-cs(c)).^2;
    end

      %assign members (image pixels)to minimum distance clusters
      [mv,ML]=min(D,[],3);

      %updat cluster centroid
      for c=1:nc
          I=(ML==c);
          cs(c)=mean(mean(im(I)));
      end

      %find maximum absolute difference between crrent and previous iteration
      %cluster centroids
      cmx=max(abs(cs-pcs));
      pcs=cs;

      t=t+1;

      %sum difference between centroid and their members and store it for
      %plotting energy minimization functions
      tsmld=[tsmld; sum(mv(:))];

end

%  assign a colour to each cluster

colors=hvs(nc);
sim=colors(ML,:);
sim=reshape(sim,row,col,3);

figure,subplot(1,2,1),imshow(im,map);
title('Input Image: pp1');
subplot(1,2,2);imshow(sim,map);
title('segmented Image:pp1')

figure;plot(tsmld,'*-b')
xlabel('Iteration');ylabel('Energy');
title('K-means energy minimization-pp1');
